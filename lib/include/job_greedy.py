# Job class to store id, deadline, and profit
class Job:
    def __init__(self, job_id, deadline, profit):
        self.job_id = job_id
        self.deadline = deadline
        self.profit = profit

def job_sequencing(jobs):
    # Step 1: Sort jobs by descending profit
    jobs.sort(key=lambda x: x.profit, reverse=True)

    # Step 2: Find the maximum deadline to create time slots
    max_deadline = max(job.deadline for job in jobs)
    slots = [-1] * (max_deadline + 1)  # 1-indexed slots

    total_profit = 0
    job_count = 0

    # Step 3: Assign jobs to slots greedily
    for job in jobs:
        # Try to find a free slot before or on the deadline
        for t in range(job.deadline, 0, -1):
            if slots[t] == -1:
                slots[t] = job.job_id
                total_profit += job.profit
                job_count += 1
                break

    # Step 4: Print result
    print("Scheduled Jobs:", [job_id for job_id in slots if job_id != -1])
    print("Total Jobs Done:", job_count)
    print("Total Profit:", total_profit)

# Example usage
jobs = [
    Job('A', 2, 100),
    Job('B', 1, 19),
    Job('C', 2, 27),
    Job('D', 1, 25),
    Job('E', 3, 15)
]

job_sequencing(jobs)
