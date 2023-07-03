package main

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha1"
	"errors"
	"flag"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/minio/minio-go"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

var flags struct {
	Endpoint     string
	AccessKey    string
	AccessSecret string
	Port         int
	Username     string
	Password     string
	expire       int
	auth         bool
}

// minio-proxy -endpoint localhost:9090 -key minioadmin -secret minioadmin -port 10086 -username admin -password admin -expire 1440 -auth false
func main() {
	flag.StringVar(&flags.Endpoint, "endpoint", "localhost:9090", "s3 endpoint")
	flag.StringVar(&flags.AccessKey, "key", "minioadmin", "s3 access key")
	flag.StringVar(&flags.AccessSecret, "secret", "minioadmin", "s3 secret key")
	flag.IntVar(&flags.Port, "port", 10085, "http server port")
	flag.StringVar(&flags.Username, "username", "admin", "http server username")
	flag.StringVar(&flags.Password, "password", "admin", "http server password")
	flag.IntVar(&flags.expire, "expire", 24*60, "http server session expire time in minutes")
	flag.BoolVar(&flags.auth, "auth", false, "enable http server auth")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
	}

	flag.Parse()
	log.Printf("flags: %+v\n", flags)

	s3Cli := NewS3Client(&S3Config{
		Endpoint:     flags.Endpoint,
		AccessKey:    flags.AccessKey,
		AccessSecret: flags.AccessSecret,
		UseSSL:       false,
	})

	err := s3Cli.Init()
	if err != nil {
		log.Fatal(err)
	}

	ws := NewWebServer(flags.Port, s3Cli)
	ws.Start()
}

const (
	AuthorizationH  = "Authorization"
	TokenCookieName = "X-Token"
)

type WebServer struct {
	port   int
	router *gin.Engine
	s3Cli  *S3Client
	cache  *LRU
}

func NewWebServer(port int, s3Cli *S3Client) *WebServer {
	return &WebServer{
		port:   port,
		router: gin.New(),
		s3Cli:  s3Cli,
		cache:  NewLRU(10, int64(flags.expire)*60),
	}
}

func (ws *WebServer) Start() {
	ws.router.Use(ws.applyAuth())

	ws.router.POST("/login", ws.handleLogin)
	ws.router.GET("/file", ws.handleLisBuckets)
	ws.router.GET("/file/:bucket", ws.handleListFiles)
	ws.router.GET("/file/:bucket/:key", ws.handleGetObject)

	ws.router.POST("/file/:bucket/:key", ws.handlePutObject)

	// Start the HTTP server
	endpoint := fmt.Sprintf(":%d", ws.port)
	log.Println("start http server on ", endpoint)
	err := ws.router.Run(endpoint)
	if err != nil {
		log.Fatal(err)
	}
}

func (ws *WebServer) applyAuth() gin.HandlerFunc {

	whitelist := []string{
		"/login",
	}

	log.Println("[middleware] apply auth, whitelist:", whitelist)

	whitelistMap := make(map[string]struct{})
	for _, url := range whitelist {
		whitelistMap[url] = struct{}{}
	}

	return func(c *gin.Context) {
		if !flags.auth {
			c.Next()
			return
		}

		// Check if the URL is in the whitelist
		url := c.Request.URL.Path
		// if match whitelist, skip auth
		if _, ok := whitelistMap[url]; ok {
			log.Printf("[middleware] url %s in whitelist, skip auth\n", url)
			c.Next()
			return
		}

		var token string

		// Check if the Authorization header is present
		token = c.GetHeader(AuthorizationH)
		if token == "" {
			// Check if the token cookie is present
			tokenCookie, err := c.Request.Cookie(TokenCookieName)
			if err != nil {
				_ = c.AbortWithError(http.StatusBadRequest, fmt.Errorf("missing token"))
				return
			}
			token = tokenCookie.Value
		}

		// Check if the token is valid
		_, ok := ws.cache.Get(token)
		if !ok {
			_ = c.AbortWithError(http.StatusUnauthorized, errors.New("invalid token"))
			return
		}

		// Renew the token
		ws.cache.Renew(token)
		ws.setupCookie(c, token)

		c.Next()
	}

}

type UserLoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func (ws *WebServer) handleLogin(c *gin.Context) {
	var req UserLoginRequest
	err := c.BindJSON(&req)
	if err != nil {
		_ = c.AbortWithError(http.StatusBadRequest, err)
		return
	}

	log.Println("login request:", req)

	if req.Username == "" || req.Password == "" {
		_ = c.AbortWithError(http.StatusBadRequest, errors.New("username or password is blank"))
		return
	}

	if req.Username != flags.Username || req.Password != flags.Password {
		_ = c.AbortWithError(http.StatusUnauthorized, errors.New("invalid username or password"))
		return
	}

	token := ws.generateToken()
	ws.cache.Set(token, "")
	ws.setupCookie(c, token)
	c.JSON(http.StatusOK, gin.H{
		"token": token,
	})
}

// generate token, uuid + timestamp + random sha1
func (ws *WebServer) generateToken() string {
	bff := make([]byte, 0, 64)
	bff = append(bff, uuid.NewString()...)                             // uuid 36
	bff = append(bff, strconv.FormatInt(time.Now().UnixNano(), 10)...) // timestamp 19
	// random int 0-1000000
	bff = append(bff, strconv.FormatUint((rand.Uint64())%1000000, 10)...) // random 7

	return fmt.Sprintf("%x", sha1.Sum(bff))
}

func (ws *WebServer) setupCookie(c *gin.Context, token string) {
	// Set the token cookie
	tokenCookie := &http.Cookie{
		Name:     TokenCookieName,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		Secure:   false,             // Set to true if using HTTPS
		MaxAge:   flags.expire * 60, // Cookie expiration time in seconds
	}
	http.SetCookie(c.Writer, tokenCookie)
}

func (ws *WebServer) handleLisBuckets(c *gin.Context) {
	log.Println("list buckets")
	buckets, err := ws.s3Cli.ListBuckets()
	if err != nil {
		_ = c.AbortWithError(500, err)
		return
	}

	log.Println("list buckets success, buckets:", buckets)

	c.JSON(200, buckets)
}

func (ws *WebServer) handleListFiles(c *gin.Context) {
	bucket := c.Param("bucket")
	if bucket == "" {
		_ = c.AbortWithError(400, errors.New("bucket name is empty"))
		return
	}
	log.Println("list bucket objects:", bucket)
	flist := ws.s3Cli.ListFiles(bucket, "")
	log.Println("list bucket objects success, objects size:", len(flist))
	c.JSON(200, flist)
}

func (ws *WebServer) handleGetObject(c *gin.Context) {
	bucket := c.Param("bucket")
	key := c.Param("key")

	log.Printf("fetch bucket: %s, object: %s\n", bucket, key)

	if bucket == "" || key == "" {
		_ = c.AbortWithError(http.StatusBadRequest, errors.New("bucket name or key is empty"))
		return
	}

	bs, err := ws.s3Cli.FetchStream(c.Request.Context(), bucket, key, minio.GetObjectOptions{})
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			log.Printf("fetch object failed, object not found: %+v", err)
			_ = c.AbortWithError(http.StatusNotFound, err)
			return
		}
		log.Printf("fetch object failed: %+v", err)
		_ = c.AbortWithError(http.StatusInternalServerError, err)
		return
	}

	if len(bs) == 0 {
		log.Printf("fetch object failed, object is empty")
		_ = c.AbortWithError(http.StatusNotFound, errors.New("object is empty"))
		return
	}

	// get magic number from bytes
	contentType := http.DetectContentType(bs)
	log.Printf("fetch object success, file size: %d\n, content-type: %s", len(bs), contentType)

	// set content type
	c.Header("Content-Type", contentType)
	c.Status(http.StatusOK)
	_, _ = c.Writer.Write(bs)
}

func (ws *WebServer) handlePutObject(c *gin.Context) {
	bucket := c.Param("bucket")
	key := c.Param("key")

	log.Printf("put into bucket: %s, object: %s\n", bucket, key)

	if bucket == "" || key == "" {
		_ = c.AbortWithError(http.StatusBadRequest, errors.New("bucket name or key is empty"))
		return
	}

	err := ws.s3Cli.CreateBucket(c.Request.Context(), bucket)
	if err != nil {
		_ = c.AbortWithError(http.StatusInternalServerError, fmt.Errorf("create bucket err: %s", err.Error()))
		return
	}

	// Single file
	file, err := c.FormFile("file")
	if err != nil {
		_ = c.AbortWithError(http.StatusBadRequest, fmt.Errorf("get form file err: %s", err.Error()))
		return
	}

	openedFile, err := file.Open()
	if err != nil {
		_ = c.AbortWithError(http.StatusBadRequest, fmt.Errorf("open file err: %s", err.Error()))
		return
	}
	defer openedFile.Close()

	// Read the file into a byte array
	bs, err := io.ReadAll(openedFile)
	if err != nil {
		_ = c.AbortWithError(http.StatusBadRequest, fmt.Errorf("read file err: %s", err.Error()))
		return
	}

	contentType := http.DetectContentType(bs)
	opts := minio.PutObjectOptions{
		ContentType: contentType,
	}

	err = ws.s3Cli.UploadFileFromBytes(c.Request.Context(), bucket, key, bs, opts)
	if err != nil {
		_ = c.AbortWithError(http.StatusInternalServerError, fmt.Errorf("upload file err: %s", err.Error()))
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": fmt.Sprintf("'%s' uploaded!", file.Filename),
	})

}

type S3Config struct {
	Endpoint     string `yaml:"endpoint"`
	AccessKey    string `yaml:"access_key"`
	AccessSecret string `yaml:"access_secret"`
	UseSSL       bool   `yaml:"use_ssl"`
}

var (
	ErrNotFound = errors.New("not found")
)

type S3Client struct {
	s3conf      *S3Config
	minioClient *minio.Client
}

func NewS3Client(s3conf *S3Config) *S3Client {
	return &S3Client{
		s3conf: s3conf,
	}
}

func (sc *S3Client) Init() (err error) {
	log.Printf("[S3Client] init s3 client: %+v", sc.s3conf)
	sc.minioClient, err = minio.New(sc.s3conf.Endpoint, sc.s3conf.AccessKey,
		sc.s3conf.AccessSecret, sc.s3conf.UseSSL)

	if err != nil {
		return err
	}

	// test s3 auth
	if _, err = sc.minioClient.ListBuckets(); err != nil {
		return err
	}

	return nil
}

func (sc *S3Client) ListBuckets() (lst []string, err error) {
	buckets, err := sc.minioClient.ListBuckets()
	if err != nil {
		return nil, err
	}

	if len(buckets) == 0 {
		return []string{}, nil
	}

	lst = make([]string, 0)

	for _, obj := range buckets {
		lst = append(lst, obj.Name)
	}

	return lst, nil
}

func (sc *S3Client) ListFiles(bucket, objectPrefix string) (lst []string) {
	lst = []string{}

	objectCh := sc.minioClient.ListObjectsV2(bucket, objectPrefix, true, nil)
	for object := range objectCh {
		if object.Err != nil {
			log.Println("query file list failed")
			continue
		}

		remoteFile := object.Key
		lst = append(lst, remoteFile)
	}
	return lst
}

func (sc *S3Client) Fetch(ctx context.Context, bucket, objectName, localFile string,
	opts minio.GetObjectOptions) error {
	err := sc.minioClient.FGetObjectWithContext(ctx, bucket, objectName, localFile, opts)
	if err != nil {
		var respErr minio.ErrorResponse
		if errors.As(err, &respErr) {
			switch respErr.StatusCode {
			case http.StatusNotFound:
				return errors.Join(ErrNotFound, fmt.Errorf("object not found: %w", err))
			}
		}
		return fmt.Errorf("get object failed: %w", err)
	}

	return nil
}

func (sc *S3Client) FetchStream(ctx context.Context, bucket, objectName string,
	opts minio.GetObjectOptions) ([]byte, error) {
	obj, err := sc.minioClient.GetObjectWithContext(ctx, bucket, objectName, opts)
	if err != nil {
		return nil, fmt.Errorf("get object failed: %w", err)
	}
	defer obj.Close()

	objStat, err := obj.Stat()
	if err != nil {
		var respErr minio.ErrorResponse
		if errors.As(err, &respErr) {
			switch respErr.StatusCode {
			case http.StatusNotFound:
				return nil, errors.Join(ErrNotFound, fmt.Errorf("object not found: %w", err))
			}
		}

		return nil, fmt.Errorf("get object stat failed: %w", err)
	}

	if objStat.Size == 0 {
		return nil, fmt.Errorf("object size is 0")
	}

	buff := bytes.NewBuffer(make([]byte, 0, objStat.Size))
	for {
		bs := make([]byte, 8192)
		n, err := obj.Read(bs)
		if err != nil {
			if errors.Is(err, io.EOF) {
				buff.Write(bs[:n])
				break
			} else {
				return nil, fmt.Errorf("read object failed: %w", err)
			}
		}
		buff.Write(bs[:n])
	}

	return buff.Bytes(), nil
}

// UploadFile uploads a file to s3
func (sc *S3Client) UploadFile(ctx context.Context, bucket, objectName, localFile string,
	opts minio.PutObjectOptions) error {
	_, err := sc.minioClient.FPutObjectWithContext(ctx, bucket, objectName, localFile, opts)
	if err != nil {
		return err
	}

	return nil
}

// UploadFileFromStream uploads a file from stream to s3
func (sc *S3Client) UploadFileFromStream(ctx context.Context, bucket, objectName string, reader io.Reader, size int64,
	opts minio.PutObjectOptions) error {
	_, err := sc.minioClient.PutObjectWithContext(ctx, bucket, objectName, reader, size, opts)
	if err != nil {
		return err
	}
	return nil
}

// UploadFileFromBytes uploads a file from bytes to s3
func (sc *S3Client) UploadFileFromBytes(ctx context.Context, bucket, objectName string, bs []byte,
	opts minio.PutObjectOptions) error {
	reader := bytes.NewReader(bs)
	return sc.UploadFileFromStream(ctx, bucket, objectName, reader, int64(len(bs)), opts)
}

// CreateBucket creates a bucket if not exist
func (sc *S3Client) CreateBucket(ctx context.Context, bucket string) error {
	if exist, err := sc.minioClient.BucketExists(bucket); err != nil {
		return err
	} else if exist {
		return nil
	} else {
		return sc.minioClient.MakeBucket(bucket, "")
	}
}

// entry is used to hold a value in the queue
type entry struct {
	key       string
	value     string
	timestamp int64 // UNIX timestamp
}

type LRU struct {
	capacity int
	cache    map[string]*list.Element
	queue    *list.List
	lock     sync.RWMutex
	ttl      int64 // Time To Live in seconds
}

func NewLRU(capacity int, ttl int64) *LRU {
	lru := &LRU{
		capacity: capacity,
		cache:    make(map[string]*list.Element),
		queue:    list.New(),
		ttl:      ttl,
	}

	go lru.purgeRoutine()

	return lru
}

func (l *LRU) purgeRoutine() {
	for {
		l.lock.Lock()
		for key, el := range l.cache {
			e := el.Value.(*entry)
			if time.Now().Unix()-e.timestamp > l.ttl {
				l.queue.Remove(el)
				delete(l.cache, key)
			}
		}
		l.lock.Unlock()
		time.Sleep(5 * time.Minute)
	}
}

func (l *LRU) Get(key string) (string, bool) {
	l.lock.RLock()
	el, ok := l.cache[key]
	l.lock.RUnlock()

	if !ok {
		return "", false
	}

	l.lock.Lock()
	l.queue.MoveToFront(el)
	e := el.Value.(*entry)
	l.lock.Unlock()
	return e.value, true
}

func (l *LRU) Set(key string, value string) {
	l.lock.Lock()
	defer l.lock.Unlock()

	if el, ok := l.cache[key]; ok {
		l.queue.MoveToFront(el)
		e := el.Value.(*entry)
		e.value = value
		e.timestamp = time.Now().Unix()
	} else {
		if l.queue.Len() >= l.capacity {
			el := l.queue.Back()
			delete(l.cache, el.Value.(*entry).key)
			l.queue.Remove(el)
		}
		e := &entry{key, value, time.Now().Unix()}
		el := l.queue.PushFront(e)
		l.cache[key] = el
	}
}

// Del deletes a key from the cache
func (l *LRU) Del(key string) {
	l.lock.Lock()
	defer l.lock.Unlock()

	if el, ok := l.cache[key]; ok {
		delete(l.cache, el.Value.(*entry).key)
		l.queue.Remove(el)
	}
}

// Renew resets the TTL of a key
func (l *LRU) Renew(key string) {
	l.lock.Lock()
	defer l.lock.Unlock()

	if el, ok := l.cache[key]; ok {
		e := el.Value.(*entry)
		e.timestamp = time.Now().Unix()
	}
}
