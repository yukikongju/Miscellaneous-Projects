#  https://leetcode.com/problems/number-of-zero-filled-subarrays/description/

class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        # solution: sliding window O(n)

        count = 0
        left, right = 0, 0

        while (right < len(nums)):
            
            if nums[right] == 0:
                count += right - left + 1
                right += 1
            else:
                right += 1
                left = right

        return count
        
