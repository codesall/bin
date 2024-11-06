import java.util.*;

public class FCFS {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("ENTER THE NUMBER OF PROCESSES:");
        int n = sc.nextInt();

        int PID[] = new int[n]; // Process IDs
        int AT[] = new int[n]; // Arrival Times
        int BT[] = new int[n]; // Burst Times
        int CT[] = new int[n]; // Completion Times
        int TAT[] = new int[n]; // Turnaround Times
        int WT[] = new int[n]; // Waiting Times

        // Input process details
        for (int i = 0; i < n; i++) {
            System.out.println("Enter the Process ID:");
            PID[i] = sc.nextInt();
            System.out.println("Enter the Arrival Time:");
            AT[i] = sc.nextInt();
            System.out.println("Enter the Burst Time:");
            BT[i] = sc.nextInt();
        }

        // Sort processes by Arrival Time using a simple bubble sort
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                if (AT[j] > AT[j + 1]) {
                    // Swap Arrival Times
                    int temp = AT[j];
                    AT[j] = AT[j + 1];
                    AT[j + 1] = temp;

                    // Swap Burst Times
                    temp = BT[j];
                    BT[j] = BT[j + 1];
                    BT[j + 1] = temp;

                    // Swap Process IDs
                    temp = PID[j];
                    PID[j] = PID[j + 1];
                    PID[j + 1] = temp;
                }
            }
        }

        // Calculate Completion Times
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                CT[i] = AT[i] + BT[i]; // First process completion time
            } else {
                // Check if the process arrives after the previous one has completed
                if (AT[i] > CT[i - 1]) {
                    CT[i] = AT[i] + BT[i]; // Wait for arrival
                } else {
                    CT[i] = CT[i - 1] + BT[i]; // Continue processing
                }
            }
        }

        // Calculate Turnaround and Waiting Times
        for (int i = 0; i < n; i++) {
            TAT[i] = CT[i] - AT[i]; // Turnaround Time
            WT[i] = TAT[i] - BT[i]; // Waiting Time
        }

        // Display the results
        System.out.println("Process\tAT\tBT\tCT\tTAT\tWT");
        for (int i = 0; i < n; i++) {
            System.out.println(PID[i] + "\t" + AT[i] + "\t" + BT[i] + "\t" + CT[i] + "\t" + TAT[i] + "\t" + WT[i]);
        }

        // Calculate averages
        float sumTAT = 0, sumWT = 0;
        for (int i = 0; i < n; i++) {
            sumTAT += TAT[i];
            sumWT += WT[i];
        }

        float avgTAT = sumTAT / n; // Average Turnaround Time
        float avgWT = sumWT / n; // Average Waiting Time

        // Print average times
        System.out.printf("Average Turnaround Time: %.2f\n", avgTAT);
        System.out.printf("Average Waiting Time: %.2f\n", avgWT);

        // Close the scanner
        sc.close();
    }
}
