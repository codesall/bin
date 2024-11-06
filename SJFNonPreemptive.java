import java.util.*;

class SJFNonPreemptive {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the number of processes: ");
        int numProcesses = scanner.nextInt();

        int processId[] = new int[numProcesses];
        int arrivalTime[] = new int[numProcesses];
        int burstTime[] = new int[numProcesses];
        int completionTime[] = new int[numProcesses];
        int turnaroundTime[] = new int[numProcesses];
        int waitingTime[] = new int[numProcesses];
        int finished[] = new int[numProcesses];

        for (int i = 0; i < numProcesses; i++) {
            System.out.println("Enter process ID: ");
            processId[i] = scanner.nextInt();

            System.out.println("Enter Arrival time: ");
            arrivalTime[i] = scanner.nextInt();

            System.out.println("Enter Burst time: ");
            burstTime[i] = scanner.nextInt();

            finished[i] = 0; // Initially, all processes are incomplete
        }
        scanner.close();

        int systemTime = 0; // System time
        int completedProcesses = 0; // Count of completed processes

        while (true) {
            if (completedProcesses == numProcesses)
                break;

            int selectedProcessIndex = numProcesses; // Index of the selected process
            int minBurstTime = Integer.MAX_VALUE; // Minimum burst time found

            for (int i = 0; i < numProcesses; i++) {
                if (arrivalTime[i] <= systemTime && finished[i] == 0 && burstTime[i] < minBurstTime) {
                    selectedProcessIndex = i;
                    minBurstTime = burstTime[i];
                }
            }

            if (selectedProcessIndex == numProcesses) {
                systemTime++; // No process is ready; increment system time
            } else {
                completionTime[selectedProcessIndex] = systemTime + burstTime[selectedProcessIndex];// Calculate
                                                                                                    // completion time
                finished[selectedProcessIndex] = 1; // Mark process as completed
                systemTime = completionTime[selectedProcessIndex]; // Update system time
                completedProcesses++; // Increment the count of completed processes
            }
        }

        // Calculate Turnaround Time (TAT) and Waiting Time (WT) for each process
        float totalTurnaroundTime = 0, totalWaitingTime = 0;
        for (int k = 0; k < numProcesses; k++) {
            turnaroundTime[k] = completionTime[k] - arrivalTime[k];
            waitingTime[k] = turnaroundTime[k] - burstTime[k];
            totalTurnaroundTime += turnaroundTime[k];
            totalWaitingTime += waitingTime[k];
        }

        // Display process details
        System.out.println("PID\tAT\tBT\tCT\tTAT\tWT");
        for (int j = 0; j < numProcesses; j++) {
            System.out.println(processId[j] + "\t" + arrivalTime[j] + "\t" + burstTime[j] + "\t" + completionTime[j]
                    + "\t" + turnaroundTime[j] + "\t" + waitingTime[j]);
        }

        // Calculate and display average TAT and WT
        float avgTurnaroundTime = totalTurnaroundTime / numProcesses;
        float avgWaitingTime = totalWaitingTime / numProcesses;
        System.out.println("Average Turnaround Time (TAT): " + avgTurnaroundTime);
        System.out.println("Average Waiting Time (WT): " + avgWaitingTime);
    }
}
