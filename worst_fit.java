import java.util.Scanner;

public class worst_fit {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the number of memory blocks: ");
        int numBlocks = scanner.nextInt();
        int[] memoryBlocks = new int[numBlocks];
        boolean[] isBlockUsed = new boolean[numBlocks];

        System.out.println("Enter the size of each memory block:");
        for (int i = 0; i < numBlocks; i++) {
            memoryBlocks[i] = scanner.nextInt();
        }

        System.out.print("Enter the number of processes: ");
        int numProcesses = scanner.nextInt();
        int[] processSizes = new int[numProcesses];

        System.out.println("Enter the size of each process:");
        for (int i = 0; i < numProcesses; i++) {
            processSizes[i] = scanner.nextInt();
        }

        for (int i = 0; i < numProcesses; i++) {
            int worstIndex = -1;
            for (int j = 0; j < numBlocks; j++) {

                if (!isBlockUsed[j] && memoryBlocks[j] >= processSizes[i]) {

                    if (worstIndex == -1 || memoryBlocks[worstIndex] < memoryBlocks[j]) {
                        worstIndex = j;
                    }
                }
            }

            if (worstIndex != -1) {
                System.out.println("Process " + (i + 1) + " of size " + processSizes[i] + " allocated to block "
                        + (worstIndex + 1));
                memoryBlocks[worstIndex] -= processSizes[i];
                isBlockUsed[worstIndex] = true;
            } else {
                System.out.println("Process " + (i + 1) + " of size " + processSizes[i] + " could not be allocated.");
            }
        }

        scanner.close();
    }
}
