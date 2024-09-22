#  https://leetcode.com/problems/length-of-the-longest-alphabetical-continuous-substring/description/

class Solution:
    def longestContinuousSubstring(self, s: str) -> int:
        # solution: sliding window O(n)

        # base case
        if len(s) <= 1:
            return 1

        longest = 1
        left, right = 0, 1

        while (right < len(s)):
            letter, prev_letter = s[right], s[right-1]
            if ord(letter) == ord(prev_letter) + 1:
                longest = max(longest, right - left + 1)
            else:
                left = right
            right += 1

        return longest

        
