import java.util.Scanner;

public class BestFit {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Initialize memory blocks
        System.out.print("Enter the number of memory blocks: ");
        int numBlocks = scanner.nextInt();
        int[] memoryBlocks = new int[numBlocks];

        System.out.println("Enter the sizes of the memory blocks:");
        for (int i = 0; i < numBlocks; i++) {
            System.out.print("Block " + (i + 1) + ": ");
            memoryBlocks[i] = scanner.nextInt();
        }

        // Initialize processes
        System.out.print("\nEnter the number of processes: ");
        int numProcesses = scanner.nextInt();
        int[] processSizes = new int[numProcesses];

        System.out.println("Enter the sizes of the processes:");
        for (int i = 0; i < numProcesses; i++) {
            System.out.print("Process " + (i + 1) + ": ");
            processSizes[i] = scanner.nextInt();
        }

        // Best Fit Allocation
        int[] allocation = new int[numProcesses];

        // Initialize all allocations to -1 (indicating unallocated)
        for (int i = 0; i < allocation.length; i++) {
            allocation[i] = -1;
        }

        // Process each process one by one
        for (int i = 0; i < numProcesses; i++) {
            int bestIndex = -1;

            for (int j = 0; j < numBlocks; j++) {
                if (memoryBlocks[j] >= processSizes[i]) {
                    if (bestIndex == -1 || memoryBlocks[j] < memoryBlocks[bestIndex]) {
                        bestIndex = j;
                    }
                }
            }

            // If a suitable block was found
            if (bestIndex != -1) {
                allocation[i] = bestIndex;
                memoryBlocks[bestIndex] -= processSizes[i];
            }
        }

        // Display the allocation results
        System.out.println("\nProcess No.\tProcess Size\tBlock No.");
        for (int i = 0; i < numProcesses; i++) {
            System.out.print(" " + (i + 1) + "\t\t" + processSizes[i] + "\t\t");
            if (allocation[i] != -1) {
                System.out.println((allocation[i] + 1));
            } else {
                System.out.println("Not Allocated");
            }
        }

        scanner.close();
    }
}
