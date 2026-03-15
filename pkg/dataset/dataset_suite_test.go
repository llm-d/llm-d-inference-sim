package dataset

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

var tokenizerMngr *tokenizer.TokenizerManager

func TestDataset(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Dataset Suite")
}

var _ = BeforeSuite(func() {
	tokenizerMngr = tokenizer.NewTokenizerManager()
	err := tokenizerMngr.Init(context.Background(), klog.Background())
	Expect(err).ShouldNot(HaveOccurred())
})

var _ = AfterSuite(func() {
	tokenizerMngr.Clean()
})
