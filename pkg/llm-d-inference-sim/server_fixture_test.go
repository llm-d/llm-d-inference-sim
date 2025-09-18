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
	"os"
	"path/filepath"
)

// GenerateTempCerts creates temporary SSL certificate and key files for testing
func GenerateTempCerts(tempDir string) (certFile, keyFile string, err error) {
	certPEM, keyPEM, err := CreateSelfSignedTLSCertificatePEM()
	if err != nil {
		return "", "", err
	}

	certFile = filepath.Join(tempDir, "cert.pem")
	if err := os.WriteFile(certFile, certPEM, 0644); err != nil {
		return "", "", err
	}

	keyFile = filepath.Join(tempDir, "key.pem")
	if err := os.WriteFile(keyFile, keyPEM, 0600); err != nil {
		return "", "", err
	}

	return certFile, keyFile, nil
}
