def fractional_knapsack(): 
    weights = [10, 20, 30] 
    values = [60, 100, 120] 
    capacity = 50 
    res = 0 

    # Pair each item's (weight, value)
    for pair in sorted(zip(weights, values), key=lambda x: x[1]/x[0], reverse=True): 
        if capacity <= 0:  # If knapsack is full
            break  

        if pair[0] > capacity:  # Take fraction of the current item
            res += capacity * (pair[1] / pair[0])  
            capacity = 0  # Knapsack full
        else:  # Take the whole item
            res += pair[1] 
            capacity -= pair[0] 

    print("Maximum value in Knapsack =", res)
 
if __name__ == "__main__": 
    fractional_knapsack()

