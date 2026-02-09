package dataset

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

func getInputHash(tokens []uint32) []byte {
	hasher := sha256.New()

	buf := make([]byte, 4)
	for _, tokenID := range tokens {
		binary.LittleEndian.PutUint32(buf, tokenID)
		hasher.Write(buf)
	}

	// Get the hash sum as a byte slice
	return hasher.Sum(nil)
}

// validateFileNotExists checks if an output database file already exists at the given path
// Returns an error if the file exists or if there's an issue checking the file
func validateFileNotExists(path string) error {
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("output file already exists: %s", path)
	} else if !os.IsNotExist(err) {
		// Some other error occurred (permissions, etc.)
		return fmt.Errorf("error checking output file: %w", err)
	}
	return nil
}

// parseSourceJson parses the given json to array of datasetRecord
func parseSourceJson(data []byte) ([]datasetRecord, error) {
	var records []datasetRecord

	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("unmarshal: %v", err)
	}

	return records, nil
}

// loadLocalFile loads file
func loadLocalFile(fullPath string) ([]byte, error) {
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return nil, errors.Join(err, fmt.Errorf("failed to read file %s", fullPath))
	}
	return data, nil
}
