/*
Copyright 2025 The llm-d-inference-sim Authors.

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

package common

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func createTestCertificate(serialNum int64) (certPEM, keyPEM []byte, err error) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	template := x509.Certificate{
		SerialNumber: big.NewInt(serialNum),
		Subject: pkix.Name{
			Organization: []string{"Test Org"},
			CommonName:   "test-cert",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})

	privBytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal private key: %w", err)
	}
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes})

	return certPEM, keyPEM, nil
}

func writeCertFiles(dir, certName, keyName string, certPEM, keyPEM []byte) (certFile, keyFile string) {
	certFile = filepath.Join(dir, certName)
	keyFile = filepath.Join(dir, keyName)
	ExpectWithOffset(1, os.WriteFile(certFile, certPEM, 0644)).To(Succeed())
	ExpectWithOffset(1, os.WriteFile(keyFile, keyPEM, 0600)).To(Succeed())
	return certFile, keyFile
}

func setupK8sStyleCertDir(baseDir string, certPEM, keyPEM []byte) {
	timestamp := time.Now().Format("..2006_01_02_15_04_05.000000000")
	dataDir := filepath.Join(baseDir, timestamp)
	Expect(os.MkdirAll(dataDir, 0755)).To(Succeed())

	Expect(os.WriteFile(filepath.Join(dataDir, "tls.crt"), certPEM, 0644)).To(Succeed())
	Expect(os.WriteFile(filepath.Join(dataDir, "tls.key"), keyPEM, 0600)).To(Succeed())

	Expect(os.Symlink(timestamp, filepath.Join(baseDir, "..data"))).To(Succeed())
	Expect(os.Symlink(filepath.Join("..data", "tls.crt"), filepath.Join(baseDir, "tls.crt"))).To(Succeed())
	Expect(os.Symlink(filepath.Join("..data", "tls.key"), filepath.Join(baseDir, "tls.key"))).To(Succeed())
}

func updateK8sStyleCerts(baseDir string, certPEM, keyPEM []byte) {
	newTimestamp := time.Now().Format("..2006_01_02_15_04_05.000000000")
	newDataDir := filepath.Join(baseDir, newTimestamp)
	Expect(os.MkdirAll(newDataDir, 0755)).To(Succeed())

	Expect(os.WriteFile(filepath.Join(newDataDir, "tls.crt"), certPEM, 0644)).To(Succeed())
	Expect(os.WriteFile(filepath.Join(newDataDir, "tls.key"), keyPEM, 0600)).To(Succeed())

	dotDataTmp := filepath.Join(baseDir, "..data_tmp")
	Expect(os.Symlink(newTimestamp, dotDataTmp)).To(Succeed())
	Expect(os.Rename(dotDataTmp, filepath.Join(baseDir, "..data"))).To(Succeed())
}

func certSerialNumber(cert *tls.Certificate) int64 {
	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	ExpectWithOffset(1, err).NotTo(HaveOccurred())
	return x509Cert.SerialNumber.Int64()
}

var _ = Describe("CertReloader", func() {

	It("should load the initial certificate", func() {
		certPEM, keyPEM, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		certFile, keyFile := writeCertFiles(dir, "tls.crt", "tls.key", certPEM, keyPEM)

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())

		loaded := reloader.Get()
		Expect(loaded).NotTo(BeNil())
		Expect(certSerialNumber(loaded)).To(Equal(int64(1)))
	})

	It("should reload when certificate files are updated", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		certFile, keyFile := writeCertFiles(dir, "tls.crt", "tls.key", certPEM1, keyPEM1)

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())
		Expect(certSerialNumber(reloader.Get())).To(Equal(int64(1)))

		certPEM2, keyPEM2, err := createTestCertificate(2)
		Expect(err).NotTo(HaveOccurred())

		// Atomic rename to avoid partial reads.
		Expect(os.WriteFile(certFile+".tmp", certPEM2, 0644)).To(Succeed())
		Expect(os.WriteFile(keyFile+".tmp", keyPEM2, 0600)).To(Succeed())
		Expect(os.Rename(keyFile+".tmp", keyFile)).To(Succeed())
		Expect(os.Rename(certFile+".tmp", certFile)).To(Succeed())

		Eventually(func() int64 {
			return certSerialNumber(reloader.Get())
		}).WithTimeout(5 * time.Second).WithPolling(50 * time.Millisecond).Should(Equal(int64(2)))
	})

	It("should handle multiple sequential updates", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		certFile, keyFile := writeCertFiles(dir, "tls.crt", "tls.key", certPEM1, keyPEM1)

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())

		for i := int64(2); i <= 5; i++ {
			certPEM, keyPEM, err := createTestCertificate(i)
			Expect(err).NotTo(HaveOccurred())

			Expect(os.WriteFile(certFile+".tmp", certPEM, 0644)).To(Succeed())
			Expect(os.WriteFile(keyFile+".tmp", keyPEM, 0600)).To(Succeed())
			Expect(os.Rename(keyFile+".tmp", keyFile)).To(Succeed())
			Expect(os.Rename(certFile+".tmp", certFile)).To(Succeed())

			expected := i
			Eventually(func() int64 {
				return certSerialNumber(reloader.Get())
			}).WithTimeout(5 * time.Second).WithPolling(50 * time.Millisecond).Should(Equal(expected))
		}
	})

	It("should keep the old certificate on reload error", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		certFile, keyFile := writeCertFiles(dir, "tls.crt", "tls.key", certPEM1, keyPEM1)

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())

		// Write mismatched cert and key.
		certPEM2, _, err := createTestCertificate(2)
		Expect(err).NotTo(HaveOccurred())
		_, keyPEM3, err := createTestCertificate(3)
		Expect(err).NotTo(HaveOccurred())

		Expect(os.WriteFile(certFile, certPEM2, 0644)).To(Succeed())
		Expect(os.WriteFile(keyFile, keyPEM3, 0600)).To(Succeed())

		// Give the reloader time to attempt (and fail) the reload.
		Consistently(func() int64 {
			return certSerialNumber(reloader.Get())
		}).WithTimeout(2 * time.Second).WithPolling(100 * time.Millisecond).Should(Equal(int64(1)))
	})

	It("should work with arbitrary filenames", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		certFile, keyFile := writeCertFiles(dir, "server.crt", "server.key", certPEM1, keyPEM1)

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())
		Expect(certSerialNumber(reloader.Get())).To(Equal(int64(1)))

		certPEM2, keyPEM2, err := createTestCertificate(2)
		Expect(err).NotTo(HaveOccurred())

		Expect(os.WriteFile(certFile+".tmp", certPEM2, 0644)).To(Succeed())
		Expect(os.WriteFile(keyFile+".tmp", keyPEM2, 0600)).To(Succeed())
		Expect(os.Rename(keyFile+".tmp", keyFile)).To(Succeed())
		Expect(os.Rename(certFile+".tmp", certFile)).To(Succeed())

		Eventually(func() int64 {
			return certSerialNumber(reloader.Get())
		}).WithTimeout(5 * time.Second).WithPolling(50 * time.Millisecond).Should(Equal(int64(2)))
	})

	It("should work with cert and key in different directories", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		certDir := GinkgoT().TempDir()
		keyDir := GinkgoT().TempDir()

		certFile := filepath.Join(certDir, "cert.pem")
		keyFile := filepath.Join(keyDir, "key.pem")
		Expect(os.WriteFile(certFile, certPEM1, 0644)).To(Succeed())
		Expect(os.WriteFile(keyFile, keyPEM1, 0600)).To(Succeed())

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())
		Expect(certSerialNumber(reloader.Get())).To(Equal(int64(1)))

		certPEM2, keyPEM2, err := createTestCertificate(2)
		Expect(err).NotTo(HaveOccurred())

		Expect(os.WriteFile(certFile+".tmp", certPEM2, 0644)).To(Succeed())
		Expect(os.WriteFile(keyFile+".tmp", keyPEM2, 0600)).To(Succeed())
		Expect(os.Rename(keyFile+".tmp", keyFile)).To(Succeed())
		Expect(os.Rename(certFile+".tmp", certFile)).To(Succeed())

		Eventually(func() int64 {
			return certSerialNumber(reloader.Get())
		}).WithTimeout(5 * time.Second).WithPolling(50 * time.Millisecond).Should(Equal(int64(2)))
	})

	It("should work with Kubernetes-style symlink rotation", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		setupK8sStyleCertDir(dir, certPEM1, keyPEM1)

		certFile := filepath.Join(dir, "tls.crt")
		keyFile := filepath.Join(dir, "tls.key")

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())
		Expect(certSerialNumber(reloader.Get())).To(Equal(int64(1)))

		certPEM2, keyPEM2, err := createTestCertificate(2)
		Expect(err).NotTo(HaveOccurred())

		updateK8sStyleCerts(dir, certPEM2, keyPEM2)

		Eventually(func() int64 {
			return certSerialNumber(reloader.Get())
		}).WithTimeout(5 * time.Second).WithPolling(50 * time.Millisecond).Should(Equal(int64(2)))
	})

	It("should stop watching when context is cancelled", func() {
		certPEM1, keyPEM1, err := createTestCertificate(1)
		Expect(err).NotTo(HaveOccurred())

		dir := GinkgoT().TempDir()
		certFile, keyFile := writeCertFiles(dir, "tls.crt", "tls.key", certPEM1, keyPEM1)

		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		Expect(err).NotTo(HaveOccurred())

		ctx, cancel := context.WithCancel(context.Background())

		reloader, err := NewCertReloader(ctx, certFile, keyFile, &cert)
		Expect(err).NotTo(HaveOccurred())

		cancel()

		// After cancel, updates should not be picked up.
		certPEM2, keyPEM2, err := createTestCertificate(2)
		Expect(err).NotTo(HaveOccurred())
		// Small sleep to let the goroutine exit.
		time.Sleep(100 * time.Millisecond)

		Expect(os.WriteFile(certFile, certPEM2, 0644)).To(Succeed())
		Expect(os.WriteFile(keyFile, keyPEM2, 0600)).To(Succeed())

		Consistently(func() int64 {
			return certSerialNumber(reloader.Get())
		}).WithTimeout(2 * time.Second).WithPolling(100 * time.Millisecond).Should(Equal(int64(1)))
	})
})
