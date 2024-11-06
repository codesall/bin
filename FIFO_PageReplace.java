import java.util.LinkedList;
import java.util.Queue;

public class FIFO_PageReplace {
    public static void main(String[] args) {
        // Input reference string and number of frames
        int[] referenceString = { 2, 3, 2, 1, 5, 2, 4, 5, 3, 2, 5, 2 };
        int numberOfFrames = 3;

        // Call FIFO page replacement method
        simulateFIFO(referenceString, numberOfFrames);
    }

    public static void simulateFIFO(int[] referenceString, int numberOfFrames) {
        // Queue to hold the frames
        Queue<Integer> frames = new LinkedList<>();
        int pageFaults = 0;

        for (int page : referenceString) {
            // Check if the page is already in the frames
            if (!frames.contains(page)) {
                // If there are empty frames, add the page
                if (frames.size() < numberOfFrames) {
                    frames.add(page);
                } else {
                    // Remove the oldest page (FIFO) and add the new page
                    frames.poll();
                    frames.add(page);
                }
                // Increment the page fault count
                pageFaults++;
            }
            // Print current state of frames
            System.out.println("Current frames: " + frames);
        }

        // Output the total number of page faults
        System.out.println("Total Page Faults: " + pageFaults);
    }
}
