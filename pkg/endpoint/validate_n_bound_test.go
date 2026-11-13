package endpoint

import (
	"encoding/json"
	"testing"
)

// A request with n above the maximum is rejected with 400 instead of being
// split into an unbounded slice of sub-requests (which crashed the process).
func TestValidateRequest_RejectsUnboundedN(t *testing.T) {
	cases := []struct {
		n       int64
		wantErr bool
	}{
		{1, false},
		{maxRequestN, false},
		{maxRequestN + 1, true},
		{1000000000000000000, true},
	}
	for _, tc := range cases {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"n":` + itoa(tc.n) + `}`)
		var req ChatCompletionsRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("n=%d unmarshal: %v", tc.n, err)
		}
		verr := validateRequest(&req)
		if tc.wantErr && verr == nil {
			t.Errorf("n=%d: expected rejection, got none", tc.n)
		}
		if !tc.wantErr && verr != nil {
			t.Errorf("n=%d: expected acceptance, got %v", tc.n, verr)
		}
	}
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	var b [20]byte
	i := len(b)
	u := n
	if neg {
		u = -u
	}
	for u > 0 {
		i--
		b[i] = byte('0' + u%10)
		u /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
