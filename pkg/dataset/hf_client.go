/*
Copyright 2026 The llm-d-inference-sim Authors.

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

package dataset

import (
	"context"
	"errors"
	"fmt"
	"net/url"

	"github.com/valyala/fasthttp"
)

type hfClient struct {
	token string
}

func newHFClient(token string) *hfClient {
	return &hfClient{token: token}
}

// TODO change to support of a one file only!
func (c *hfClient) downloadFile(ctx context.Context, repo, filePath string) ([]byte, error) {
	urlTxt := fmt.Sprintf("https://huggingface.co/datasets/%s/resolve/main/%s", repo, filePath)
	maxRedirects := 5
	origUrl, err := url.Parse(urlTxt)
	if err != nil {
		return nil, err
	}

	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)

	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	for i := 0; i < maxRedirects; i++ {
		// check for cancellation
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("download cancelled: %w", ctx.Err())
		default:
		}

		req.Reset()
		resp.Reset()

		req.Header.SetMethod("GET")
		req.URI().Update(urlTxt)

		if c.token != "" {
			req.Header.Set("Authorization", "Bearer "+c.token)
		}

		if err := fasthttp.Do(req, resp); err != nil {
			return nil, err
		}

		statusCode := resp.StatusCode()
		if statusCode == fasthttp.StatusOK {
			return resp.Body(), nil
		}

		if statusCode >= 300 && statusCode < 400 {
			// follow the redirect Location header
			location := string(resp.Header.Peek("Location"))
			if location == "" {
				return nil, errors.New("redirect without Location header")
			}
			redirectUrl, err := origUrl.Parse(location)
			if err != nil {
				return nil, fmt.Errorf("parse redirect %q: %v", location, err)
			}
			urlTxt = redirectUrl.String()
			continue
		}

		return nil, fmt.Errorf("unexpected status %d", statusCode)
	}

	return nil, fmt.Errorf("max redirects (%d) exceeded", maxRedirects)
}
