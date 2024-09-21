#  https://leetcode.com/problems/number-of-smooth-descent-periods-of-a-stock/

class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        # solution: sliding window O(n)

        count = 0
        left, right = 0, 0 
        while (right < len(prices)):
            diff = 1 if right == 0 else prices[right-1] - prices[right]

            if diff == 1:
                count += right - left + 1
            else:
                left = right
                count += 1

            right += 1
         
        return count
        
