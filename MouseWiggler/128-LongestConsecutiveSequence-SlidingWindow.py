#  https://leetcode.com/problems/longest-consecutive-sequence/

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # solution: set

        s = set(nums)
        longest = 0
        for num in nums:
            if num - 1 not in s:
                seq_length = 0
                while (num + seq_length in s):
                    seq_length += 1
                longest = max(longest, seq_length)
        
        return longest
        
