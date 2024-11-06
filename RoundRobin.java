import java.util.*;

public class RoundRobin {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter number of processes : ");
        int n = sc.nextInt();

        System.out.println("Enter quantum time : ");
        int q = sc.nextInt();

        int pid[] = new int[n];
        int at[] = new int[n];
        int bt[] = new int[n];
        int ct[] = new int[n];
        int tat[] = new int[n];
        int wt[] = new int[n];
        int rbt[] = new int[n];
        int f[] = new int[n];

        float avgwt = 0, avgta = 0;

        for (int i = 0; i < n; i++) {
            System.out.println("Enter process ID for process " + (i + 1) + ": ");
            pid[i] = sc.nextInt();

            System.out.println("Enter Arrival Time for process " + (i + 1) + ": ");
            at[i] = sc.nextInt();

            System.out.println("Enter Burst Time for process " + (i + 1) + ": ");
            bt[i] = sc.nextInt();
            rbt[i] = bt[i];
            f[i] = 0;
        }
        System.out.println("Input accepted.");

        // Sort processes based on arrival time
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                if (at[j] > at[j + 1]) {
                    int temp;

                    temp = at[j];
                    at[j] = at[j + 1];
                    at[j + 1] = temp;

                    temp = bt[j];
                    bt[j] = bt[j + 1];
                    bt[j + 1] = temp;

                    temp = pid[j];
                    pid[j] = pid[j + 1];
                    pid[j + 1] = temp;

                    temp = rbt[j];
                    rbt[j] = rbt[j + 1];
                    rbt[j + 1] = temp;
                }
            }
        }

        // Implementing Round Robin Scheduling
        int st = 0, tot = 0;
        Queue<Integer> queue = new LinkedList<>();
        int i = 0;
        while (tot < n) {
            while (i < n && at[i] <= st) {
                queue.add(i);
                i++;
            }

            if (queue.isEmpty()) {
                st++;
                continue;
            }

            int idx = queue.poll();

            if (rbt[idx] > q) {
                st += q;
                rbt[idx] -= q;
            } else {
                st += rbt[idx];
                ct[idx] = st;
                rbt[idx] = 0;
                tot++;
            }

            while (i < n && at[i] <= st) {
                queue.add(i);
                i++;
            }

            if (rbt[idx] > 0) {
                queue.add(idx);
            }
        }

        // Calculating turn around time and waiting time
        for (i = 0; i < n; i++) {
            tat[i] = ct[i] - at[i];
            wt[i] = tat[i] - bt[i];
            avgta += tat[i];
            avgwt += wt[i];
        }

        // Printing results
        System.out.println("\nProcess ID\tArrival Time\tBurst Time\tCompletion Time\tTurnaround Time\tWaiting Time");
        for (i = 0; i < n; i++) {
            System.out.println(
                    pid[i] + "\t\t" + at[i] + "\t\t" + bt[i] + "\t\t" + ct[i] + "\t\t\t" + tat[i] + "\t\t\t" + wt[i]);
        }

        // Printing average turn around time and average waiting time
        System.out.println("\nAverage Turnaround Time: " + (avgta / n));
        System.out.println("Average Waiting Time: " + (avgwt / n));

        Scanner s = null;
        s.close();
    }
}