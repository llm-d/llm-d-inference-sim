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

package llmdinferencesim

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"time"

	"github.com/valyala/fasthttp"
)

// Based on: https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/8d01161ec48d6b49cd371f179551b35da46e6fd6/internal/tls/tls.go
func (s *VllmSimulator) configureSSL(server *fasthttp.Server) error {
	if !s.config.SSLEnabled() {
		return nil
	}

	var cert tls.Certificate
	var err error

	if s.config.SSLCertFile != "" && s.config.SSLKeyFile != "" {
		s.logger.Info("HTTPS server starting with certificate files", "cert", s.config.SSLCertFile, "key", s.config.SSLKeyFile)
		cert, err = tls.LoadX509KeyPair(s.config.SSLCertFile, s.config.SSLKeyFile)
	} else if s.config.SelfSignedCerts {
		s.logger.Info("HTTPS server starting with self-signed certificate")
		cert, err = CreateSelfSignedTLSCertificate()
	}

	if err != nil {
		s.logger.Error(err, "failed to create TLS certificate")
		return err
	}

	server.TLSConfig = &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
		},
	}

	return nil
}

// CreateSelfSignedTLSCertificatePEM creates a self-signed cert and returns the PEM-encoded certificate and key bytes
func CreateSelfSignedTLSCertificatePEM() (certPEM, keyPEM []byte, err error) {
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, nil, fmt.Errorf("error creating serial number: %v", err)
	}
	now := time.Now()
	notBefore := now.UTC()
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"llm-d Inference Simulator"},
		},
		NotBefore:             notBefore,
		NotAfter:              now.Add(time.Hour * 24 * 365 * 10).UTC(), // 10 years
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	priv, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return nil, nil, fmt.Errorf("error generating key: %v", err)
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("error creating certificate: %v", err)
	}

	certBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})

	privBytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, nil, fmt.Errorf("error marshalling private key: %v", err)
	}
	keyBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes})

	return certBytes, keyBytes, nil
}

// CreateSelfSignedTLSCertificate creates a self-signed cert the server can use to serve TLS.
// Original code: https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/8d01161ec48d6b49cd371f179551b35da46e6fd6/internal/tls/tls.go
func CreateSelfSignedTLSCertificate() (tls.Certificate, error) {
	certPEM, keyPEM, err := CreateSelfSignedTLSCertificatePEM()
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.X509KeyPair(certPEM, keyPEM)
}
