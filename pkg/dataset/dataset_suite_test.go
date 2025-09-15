package dataset_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestDataset(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Dataset Suite")
}
