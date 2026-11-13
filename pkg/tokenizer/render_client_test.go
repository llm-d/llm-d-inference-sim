/*
Copyright 2026 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tokenizer

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

// keepAliveClosingServer answers the first request on each connection and keeps the connection open. When a second
// request arrives on that connection, it reads it and closes the connection without answering, as a server does when
// its keep-alive timeout fires just as the client reuses the connection.
type keepAliveClosingServer struct {
	ln    net.Listener
	conns atomic.Int32
	wg    sync.WaitGroup
}

func newKeepAliveClosingServer() *keepAliveClosingServer {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	Expect(err).NotTo(HaveOccurred())
	s := &keepAliveClosingServer{ln: ln}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			s.conns.Add(1)
			s.wg.Add(1)
			go s.serve(conn)
		}
	}()
	return s
}

func (s *keepAliveClosingServer) serve(conn net.Conn) {
	defer s.wg.Done()
	defer conn.Close() //nolint:errcheck
	r := bufio.NewReader(conn)
	for i := 0; ; i++ {
		req, err := http.ReadRequest(r)
		if err != nil {
			return
		}
		_, _ = io.Copy(io.Discard, req.Body)
		if i > 0 {
			return // the keep-alive timeout fired: close without an answer
		}
		body := `{"token_ids":[1,2,3]}`
		_, _ = fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", len(body), body)
	}
}

func (s *keepAliveClosingServer) url() string { return "http://" + s.ln.Addr().String() }

func (s *keepAliveClosingServer) close() {
	_ = s.ln.Close()
	s.wg.Wait()
}

var _ = Describe("Render client connections", func() {
	It("resends a request on a new connection when the server closes a reused one", func() {
		srv := newKeepAliveClosingServer()
		rc := newRenderClient(context.Background(), klog.Background(), srv.url(), time.Second, time.Second)
		defer srv.close()
		defer rc.client.CloseIdleConnections()

		for i := 0; i < 2; i++ {
			tokens, _, err := rc.render("/v1/completions", []byte(`{"prompt":"hi"}`), false)
			Expect(err).NotTo(HaveOccurred(), "request %d", i+1)
			Expect(tokens).To(Equal([]uint32{1, 2, 3}))
		}
		Expect(srv.conns.Load()).To(Equal(int32(2)), "the second request should be resent on a new connection")
	})

	It("closes idle connections before vLLM's default 5 second keep-alive timeout", func() {
		rc := newRenderClient(context.Background(), klog.Background(), "http://127.0.0.1:0", time.Second, time.Second)
		transport, ok := rc.client.Transport.(*http.Transport)
		Expect(ok).To(BeTrue())
		Expect(transport.IdleConnTimeout).To(BeNumerically(">", 0))
		Expect(transport.IdleConnTimeout).To(BeNumerically("<", 5*time.Second))
	})
})
