def fractional_knapsack():
    # Step 1: Take number of items
    n = int(input("Enter number of items: "))

    weights = []
    values = []

    # Step 2: Take weight and value for each item
    print("\nEnter weight and value for each item:")
    for i in range(n):
        w = float(input(f"Weight of item {i+1}: "))
        v = float(input(f"Value of item {i+1}: "))
        weights.append(w)
        values.append(v)

    # Step 3: Take knapsack capacity
    capacity = float(input("\nEnter knapsack capacity: "))

    # Step 4: Calculate value/weight ratio for each item
    ratio = []
    for i in range(n):
        ratio.append(values[i] / weights[i])

    # Step 5: Sort items manually by ratio (descending)
    for i in range(n):
        for j in range(i + 1, n):
            if ratio[i] < ratio[j]:
                # Swap ratios
                ratio[i], ratio[j] = ratio[j], ratio[i]
                # Swap weights
                weights[i], weights[j] = weights[j], weights[i]
                # Swap values
                values[i], values[j] = values[j], values[i]

    # Step 6: Apply greedy selection
    total_value = 0.0
    print("\nItem selection process:")
    for i in range(n):
        if capacity == 0:
            break
        if weights[i] <= capacity:
            total_value += values[i]
            capacity -= weights[i]
            print(f"  Took full item (weight={weights[i]}, value={values[i]})")
        else:
            fraction = capacity / weights[i]
            total_value += values[i] * fraction
            print(f"  Took {fraction*100:.1f}% of item (weight={weights[i]}, value={values[i]})")
            capacity = 0

    print(f"\nMaximum value in knapsack = {total_value:.2f}")


# Run the program
fractional_knapsack()


# Enter number of items: 3
# Enter weight and value for each item:
# Weight of item 1: 10
# Value of item 1: 60
# Weight of item 2: 20
# Value of item 2: 100
# Weight of item 3: 30
# Value of item 3: 120

# Enter knapsack capacity: 50
