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

package dataset

import "github.com/llm-d/llm-d-inference-sim/pkg/common"

const (
	responseLenMean   = 40
	responseLenStddev = 20
)

// this array defines the probabilities for the buckets to be used for the generation of number of tokens in response
var respLenBucketsProbabilities = [...]float64{0.2, 0.3, 0.2, 0.05, 0.1, 0.15}

const (
	flexBucketIndex    = 3
	maxFixedBucketSize = 20
)

type histogramHelper struct {
	cumulativeBucketsProbabilities []float64
	random                         *common.Random
}

func newHistogramHelper(random *common.Random) *histogramHelper {
	h := histogramHelper{random: random}

	h.cumulativeBucketsProbabilities = make([]float64, len(respLenBucketsProbabilities))
	sum := 0.0

	for i, val := range respLenBucketsProbabilities {
		sum += val
		h.cumulativeBucketsProbabilities[i] = sum
	}
	return &h
}

// getResponseLengthByHistogram calculates the number of tokens to be returned in a response based on the max tokens value and the pre-defined buckets.
// The response length is distributed according to the probabilities, defined in respLenBucketsProbabilities.
// The histogram contains equally sized buckets and the last special bucket, which contains only the maxTokens value.
// The last element of respLenBucketsProbabilities defines the probability of a reposnse with maxToken tokens.
// Other values define probabilities for the equally sized buckets.
// If maxToken is small (smaller than number of buckets) - the response length is randomly selected from the range [1, maxTokens]
func (hh *histogramHelper) getResponseLengthByHistogram(maxTokens int) int {
	if maxTokens <= 1 {
		return maxTokens
	}
	// maxTokens is small - no need to use the histogram of probabilities, just select a random value in the range [1, maxTokens]
	if maxTokens <= len(hh.cumulativeBucketsProbabilities) {
		res := hh.random.RandomInt(1, maxTokens)
		return res
	}

	r := hh.random.RandomFloat(0, 1)

	// check if r is in the last bucket, then maxTokens should be returned
	if r > hh.cumulativeBucketsProbabilities[len(hh.cumulativeBucketsProbabilities)-2] {
		return maxTokens
	}

	// determine which bucket to use, the bucket with a cumulative probability larger than r is the bucket to use
	// initialize bucketIndex with the last bucket to handle the case (which should not happen) when the probabilities sum is less than 1
	bucketIndex := len(hh.cumulativeBucketsProbabilities) - 1
	for i, c := range hh.cumulativeBucketsProbabilities {
		if r <= c {
			bucketIndex = i
			break
		}
	}

	// calculate the size of all of the buckets (except the special last bucket)
	start, end := hh.calcBucketBoundaries(maxTokens, bucketIndex)

	// pick uniformly within the bucket’s range
	return hh.random.RandomInt(start, end)
}

// calcBucketBoundaries calculates boundaries of a bucket with the given index.
// Maximum size for equally sized buckets is defined by maxFixedBucketSize.
// [maxFixedBucketSize*(number-of-buckets-1)+1] is the value of maxTokens for which
// division to equally size buckets will give buckets with size maxFixedBucketSize.
// If maxTokens is [maxFixedBucketSize*(number-of-buckets-1)+1] or less,
// all buckets will be of equal size, except the last bucket, which contains only one value.
// If maxTokens is higher than [maxFixedBucketSize*(number-of-buckets-1)+1],
// and flexBucketIndex is valid (between 0 and number of buckets - 1) the buckets sizes will not be equal.
// In this case, all buckets except the one at flexBucketIndex index will have size 20 (and the last is with size 1),
// and the bucket at flexBucketIndex index will 'stretch' to cover the remaining range.
func (hh *histogramHelper) calcBucketBoundaries(maxTokens int, bucketIndex int) (start int, end int) {
	maxEquallyBucketsSz := maxFixedBucketSize*(len(hh.cumulativeBucketsProbabilities)-1) + 1

	if maxTokens <= maxEquallyBucketsSz || flexBucketIndex < 0 || flexBucketIndex >= len(hh.cumulativeBucketsProbabilities)-1 {
		// create equally size buckets
		// calculate the size of all of the buckets (except the special last bucket)
		bucketSize := float64(maxTokens-1) / float64(len(hh.cumulativeBucketsProbabilities)-1)
		start = int(bucketSize*float64(bucketIndex)) + 1
		end = int(bucketSize * float64(bucketIndex+1))
	} else {
		// create non-equally sized buckets and find boundaries of the required bucket
		if bucketIndex < flexBucketIndex {
			// the relevant bucket is before the flex bucket, all buckets are of the same size (maxFixedBucketSize)
			// start is the minimum number in the required bucket
			start = maxFixedBucketSize*bucketIndex + 1
			end = maxFixedBucketSize * (bucketIndex + 1)
		} else {
			flexBucketSize := maxTokens - (maxFixedBucketSize * (len(hh.cumulativeBucketsProbabilities) - 2))

			if bucketIndex == flexBucketIndex {
				// the relevant bucket is the flex bucket
				start = int(maxFixedBucketSize*float64(bucketIndex)) + 1
				end = maxFixedBucketSize*bucketIndex + flexBucketSize
			} else {
				// the relevant bucket is one of buckets after the flex bucket
				start = int(maxFixedBucketSize*float64(bucketIndex-1)) + flexBucketSize + 1
				end = maxFixedBucketSize*bucketIndex + flexBucketSize
			}
		}
	}

	// sometimes end could be maxTokens because of rounding, change the value to maxToken-1
	if end >= maxTokens {
		end = maxTokens - 1
	}

	return start, end
}
