package communication

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestCommunication(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Communication Suite")
}
