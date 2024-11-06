import java.util.LinkedList;
import java.util.Queue;

public class LRU_PageReplace {
    public static void main(String[] args) {
        // Input reference string and number of frames
        int[] referenceString = { 2, 3, 2, 1, 5, 2, 4, 5, 3, 2, 5, 2 };
        int numberOfFrames = 3;

        // Call LRU page replacement method
        simulateLRU(referenceString, numberOfFrames);
    }

    public static void simulateLRU(int[] referenceString, int numberOfFrames) {
        Queue<Integer> frames = new LinkedList<>();
        int pageFaults = 0;

        for (int page : referenceString) {
            // Check if the page is already in the frames
            if (!frames.contains(page)) {
                // If there are empty frames, add the page
                if (frames.size() < numberOfFrames) {
                    frames.add(page);
                } else {
                    // Remove the least recently used page and add the new page
                    frames.poll();
                    frames.add(page);
                }
                pageFaults++;
            } else {
                // If the page is already in frames, remove it and re-add it
                frames.remove(page);
                frames.add(page);
            }
            // Print current state of frames
            System.out.println("Current frames: " + frames);
        }

        // Output the total number of page faults
        System.out.println("Total Page Faults: " + pageFaults);
    }
}
