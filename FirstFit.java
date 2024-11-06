import java.util.*;

public class FirstFit {

    public static class Firstfitt {
        public static void main(String[] args) {
            Scanner sc = new Scanner(System.in);
            System.out.print("Enter the number of blocks: ");
            int m = sc.nextInt();
            int[] blockSize = new int[m];

            System.out.println("Enter the block sizes:");
            for (int i = 0; i < m; i++) {
                System.out.print("Block " + (i + 1) + ": ");
                blockSize[i] = sc.nextInt();
            }

            System.out.print("Enter the number of processes: ");
            int n = sc.nextInt();
            int[] JobSize = new int[n];

            System.out.println("Enter the process sizes:");
            for (int i = 0; i < n; i++) {
                System.out.print("Process " + (i + 1) + ": ");
                JobSize[i] = sc.nextInt();
            }
            int allocation[] = new int[n];
            int memoryWastage[] = new int[n];

            for (int i = 0; i < allocation.length; i++) {
                allocation[i] = -1;
                memoryWastage[i] = -1;
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (blockSize[j] >= JobSize[i]) {
                        allocation[i] = j;
                        memoryWastage[i] = blockSize[j] - JobSize[i];
                        blockSize[j] -= JobSize[i];
                        break;
                    }
                }
            }

            System.out.println("\nProcess No.\tProcess Size\tBlock no.\tMemory Wastage");
            for (int i = 0; i < n; i++) {
                System.out.print(" " + (i + 1) + "\t\t" + JobSize[i] + "\t\t");
                if (allocation[i] != -1) {
                    System.out.print((allocation[i] + 1) + "\t\t" + memoryWastage[i]);

                } else {
                    System.out.print("Not Allocated");
                }

                System.out.println();
            }
            sc.close();
        }
    }

}