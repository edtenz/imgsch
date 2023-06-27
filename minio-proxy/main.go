package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/minio/minio-go"
	"io"
	"log"
	"os"
	"strings"
)

var flags struct {
	Endpoint     string
	AccessKey    string
	AccessSecret string
	Port         int
}

// minio-proxy -endpoint localhost:9090 -key minioadmin -secret minioadmin -port 10086
func main() {
	flag.StringVar(&flags.Endpoint, "endpoint", "localhost:9090", "s3 endpoint")
	flag.StringVar(&flags.AccessKey, "key", "minioadmin", "s3 access key")
	flag.StringVar(&flags.AccessSecret, "secret", "minioadmin", "s3 secret key")
	flag.IntVar(&flags.Port, "port", 10085, "http server port")

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

type WebServer struct {
	port   int
	router *gin.Engine
	s3Cli  *S3Client
}

func NewWebServer(port int, s3Cli *S3Client) *WebServer {
	return &WebServer{
		port:   port,
		router: gin.New(),
		s3Cli:  s3Cli,
	}
}

func (ws *WebServer) Start() {
	ws.router.GET("/api", ws.handleLisBuckets)
	ws.router.GET("/api/:bucket", ws.handleListFiles)
	ws.router.GET("/api/:bucket/:key", ws.handleGetObject)

	// Start the HTTP server
	endpoint := fmt.Sprintf(":%d", ws.port)
	log.Println("start http server on ", endpoint)
	err := ws.router.Run(endpoint)
	if err != nil {
		log.Fatal(err)
	}
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
		_ = c.AbortWithError(400, errors.New("bucket name or key is empty"))
		return
	}

	bs, err := ws.s3Cli.FetchStream(context.Background(), bucket, key)
	if err != nil {
		log.Printf("fetch object failed: %+v", err)
		_ = c.AbortWithError(500, err)
		return
	}
	log.Println("fetch object success, file size:", len(bs))

	if strings.HasSuffix(strings.ToLower(key), ".jpg") {
		c.Header("Content-Type", "image/jpeg")
		c.Status(200)
		_, _ = c.Writer.Write(bs)
	} else if strings.HasSuffix(strings.ToLower(key), ".png") {
		c.Header("Content-Type", "image/png")
		c.Status(200)
		_, _ = c.Writer.Write(bs)
	} else {
		c.Header("Content-Disposition", "attachment; filename="+key)
		c.Data(200, "application/octet-stream", bs)
	}

}

type S3Config struct {
	Endpoint     string `yaml:"endpoint"`
	AccessKey    string `yaml:"access_key"`
	AccessSecret string `yaml:"access_secret"`
	UseSSL       bool   `yaml:"use_ssl"`
}

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

func (sc *S3Client) Fetch(ctx context.Context, bucket, objectName, localFile string) error {
	if err := sc.minioClient.FGetObjectWithContext(ctx, bucket, objectName, localFile, minio.GetObjectOptions{}); err != nil {
		return err
	}

	return nil
}

func (sc *S3Client) FetchStream(ctx context.Context, bucket, objectName string) ([]byte, error) {
	obj, err := sc.minioClient.GetObjectWithContext(ctx, bucket, objectName, minio.GetObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("get object failed: %w", err)
	}
	defer obj.Close()

	objStat, err := obj.Stat()
	if err != nil {
		return nil, fmt.Errorf("get object stat failed: %w", err)
	}

	println("object size:", objStat.Size)

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
