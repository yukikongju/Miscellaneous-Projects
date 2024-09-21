#  https://leetcode.com/problems/arithmetic-slices/description/

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # solution: sliding window O(n)

        # base case:
        if len(nums) <= 2:
            return 0

        count = 0
        left, right = 0, 2

        while (right < len(nums)):

            if nums[right] - nums[right-1] == nums[right-1] - nums[right-2]:
                count += right - left - 1
                right += 1
            else:
                left += 1
                if right - left < 2:
                    right += 1

        return count
        
