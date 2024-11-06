import java.util.*;

class PriorityNonPreemptive {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the number of processes:");
        int n = scanner.nextInt();

        int processId[] = new int[n];
        int arrivalTime[] = new int[n];
        int burstTime[] = new int[n];
        int completionTime[] = new int[n];
        int turnaroundTime[] = new int[n];
        int waitingTime[] = new int[n];
        int finished[] = new int[n];
        int originalBurstTime[] = new int[n];
        int priority[] = new int[n];

        for (int i = 0; i < n; i++) {
            System.out.println("Enter the process ID:");
            processId[i] = scanner.nextInt();
            System.out.println("Enter the arrival time:");
            arrivalTime[i] = scanner.nextInt();
            System.out.println("Enter the burst time:");
            burstTime[i] = scanner.nextInt();
            originalBurstTime[i] = burstTime[i];
            System.out.println("Enter the priority:");
            priority[i] = scanner.nextInt();
            finished[i] = 0;
        }
        scanner.close();

        int currentTime = 0, completedProcesses = 0;
        while (true) {
            if (completedProcesses == n)
                break;
            int selectedProcess = n, minPriority = Integer.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                if (arrivalTime[i] <= currentTime && finished[i] == 0 && priority[i] < minPriority) {
                    selectedProcess = i;
                    minPriority = priority[i];
                }
            }
            if (selectedProcess == n) {
                currentTime++;
            } else {
                completionTime[selectedProcess] = currentTime + burstTime[selectedProcess];
                finished[selectedProcess] = 1;
                currentTime = completionTime[selectedProcess];
                completedProcesses++;
            }
        }

        int totalTurnaroundTime = 0, totalWaitingTime = 0;
        for (int i = 0; i < n; i++) {
            turnaroundTime[i] = completionTime[i] - arrivalTime[i];
            waitingTime[i] = turnaroundTime[i] - originalBurstTime[i];
            totalTurnaroundTime += turnaroundTime[i];
            totalWaitingTime += waitingTime[i];
        }

        System.out.println("PID\tAT\tBT\tPRIO\tCT\tTAT\tWT");
        for (int i = 0; i < n; i++) {
            System.out.println(processId[i] + "\t" + arrivalTime[i] + "\t" + originalBurstTime[i] + "\t" + priority[i]
                    + "\t" + completionTime[i] + "\t" + turnaroundTime[i] + "\t" + waitingTime[i]);
        }

        double avgTurnaroundTime = (double) totalTurnaroundTime / n;
        double avgWaitingTime = (double) totalWaitingTime / n;
        System.out.println("Average Turnaround Time: " + avgTurnaroundTime);
        System.out.println("Average Waiting Time: " + avgWaitingTime);
    }
}
