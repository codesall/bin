import java.util.LinkedList;
import java.util.Queue;

public class OPTIMAL_PageReplace {
    public static void main(String[] args) {
        // Input reference string and number of frames
        int[] referenceString = { 2, 3, 2, 1, 5, 2, 4, 5, 3, 2, 5, 2 };
        int numberOfFrames = 3;

        // Call Optimal page replacement method
        simulateOptimal(referenceString, numberOfFrames);
    }

    public static void simulateOptimal(int[] referenceString, int numberOfFrames) {
        Queue<Integer> frames = new LinkedList<>();
        int pageFaults = 0;

        for (int i = 0; i < referenceString.length; i++) {
            int page = referenceString[i];

            // Check if page is already in the frames
            if (!frames.contains(page)) {
                // Page fault occurs
                if (frames.size() < numberOfFrames) {
                    frames.add(page); // Add page if there is space
                } else {
                    // Find the optimal page to replace
                    int farthest = -1;
                    int pageToRemove = -1;

                    for (int framePage : frames) {
                        int nextUse = findNextUse(referenceString, i + 1, framePage);
                        if (nextUse > farthest) {
                            farthest = nextUse;
                            pageToRemove = framePage;
                        }
                    }

                    frames.remove(pageToRemove); // Remove the optimal page
                    frames.add(page); // Add the new page
                }
                pageFaults++;
            }
            // Print current state of frames
            System.out.println("Current frames: " + frames);
        }

        // Output the total number of page faults
        System.out.println("Total Page Faults: " + pageFaults);
    }

    // Helper function to find next use of a page in the reference string
    private static int findNextUse(int[] referenceString, int startIndex, int page) {
        for (int i = startIndex; i < referenceString.length; i++) {
            if (referenceString[i] == page) {
                return i;
            }
        }
        return Integer.MAX_VALUE; // Page not found in the future
    }
}
