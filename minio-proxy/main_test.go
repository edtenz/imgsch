package main

import (
	"context"
	"testing"
	"time"
)

func TestS3Client_ListBuckets(t *testing.T) {
	type fields struct {
		s3conf *S3Config
	}
	tests := []struct {
		name   string
		fields fields
	}{
		{
			name: "test list buckets",
			fields: fields{
				s3conf: &S3Config{
					Endpoint:     "localhost:9090",
					AccessKey:    "minioadmin",
					AccessSecret: "minioadmin",
					UseSSL:       false,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sc := NewS3Client(tt.fields.s3conf)
			err := sc.Init()
			if err != nil {
				t.Errorf("Init() error = %v", err)
				return
			}
			gotLst, err := sc.ListBuckets()
			if err != nil {
				t.Errorf("ListBuckets() error = %v", err)
				return
			}
			t.Logf("got list: %+v", gotLst)
		})
	}
}

func TestS3Client_FetchStream(t *testing.T) {
	t.Run("test fetch stream", func(t *testing.T) {
		s3Cli := NewS3Client(&S3Config{
			Endpoint:     "localhost:9090",
			AccessKey:    "minioadmin",
			AccessSecret: "minioadmin",
			UseSSL:       false,
		})
		err := s3Cli.Init()
		if err != nil {
			t.Errorf("init s3 client error, %+v", err)
			return
		}

		bs, err := s3Cli.FetchStream(context.Background(), "imgsch", "002de79efc060d258b033f2509a9b5cb.jpga")
		if err != nil {
			t.Errorf("fetch stream error, %+v", err)
			return
		}

		t.Logf("fetch stream success, file size: %d", len(bs))
	})
}

func TestLRU_Set(t *testing.T) {
	t.Run("test lru set", func(t *testing.T) {
		cache := NewLRU(2, 60) // Capacity of 2, items live for 60 seconds
		cache.Set("hello", "world")
		value, ok := cache.Get("hello")
		if ok {
			t.Logf("get value: %s", value)
		}
	})

	t.Run("test lru set expire", func(t *testing.T) {
		cache := NewLRU(2, 1) // Capacity of 2, items live for 60 seconds
		cache.Set("hello", "world")
		time.Sleep(2 * time.Second)
		value, ok := cache.Get("hello")
		t.Logf("get value: %s, ok: %t", value, ok)

	})
}
