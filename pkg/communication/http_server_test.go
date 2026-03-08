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

package communication

import (
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const testModel = "test-model"

var _ = Describe("Server", func() {

	Context("SSL/HTTPS Configuration", func() {
		It("Should parse SSL certificate configuration correctly", func() {
			tempDir := GinkgoT().TempDir()
			certFile, keyFile, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
			}()

			os.Args = []string{"cmd", "--model", testModel, "--ssl-certfile", certFile, "--ssl-keyfile", keyFile}
			config, err := common.ParseCommandParamsAndLoadConfig()
			Expect(err).NotTo(HaveOccurred())
			Expect(config.SSLEnabled()).To(BeTrue())
			Expect(config.SSLCertFile).To(Equal(certFile))
			Expect(config.SSLKeyFile).To(Equal(keyFile))
		})

		It("Should parse self-signed certificate configuration correctly", func() {
			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
			}()

			os.Args = []string{"cmd", "--model", testModel, "--self-signed-certs"}
			config, err := common.ParseCommandParamsAndLoadConfig()
			Expect(err).NotTo(HaveOccurred())
			Expect(config.SSLEnabled()).To(BeTrue())
			Expect(config.SelfSignedCerts).To(BeTrue())
		})

		It("Should create self-signed TLS certificate successfully", func() {
			cert, err := CreateSelfSignedTLSCertificate()
			Expect(err).NotTo(HaveOccurred())
			Expect(cert.Certificate).To(HaveLen(1))
			Expect(cert.PrivateKey).NotTo(BeNil())
		})

		It("Should validate SSL configuration - both cert and key required", func() {
			tempDir := GinkgoT().TempDir()

			oldArgs := os.Args
			defer func() {
				os.Args = oldArgs
			}()

			certFile, _, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			os.Args = []string{"cmd", "--model", testModel, "--ssl-certfile", certFile}
			_, err = common.ParseCommandParamsAndLoadConfig()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("both ssl-certfile and ssl-keyfile must be provided together"))

			_, keyFile, err := GenerateTempCerts(tempDir)
			Expect(err).NotTo(HaveOccurred())

			os.Args = []string{"cmd", "--model", testModel, "--ssl-keyfile", keyFile}
			_, err = common.ParseCommandParamsAndLoadConfig()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("both ssl-certfile and ssl-keyfile must be provided together"))
		})
	})
})
