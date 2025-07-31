package kvcache_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestKvCache(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "KvCache Suite")
}
