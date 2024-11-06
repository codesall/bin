public class MathOperations {
    // Declare the native method that will be implemented in the DLL
    public native int add(int a, int b);

    // Load the DLL
    static {
        System.loadLibrary("MathOperations");
    }

    // Test the add function
    public static void main(String[] args) {
        MathOperations mo = new MathOperations();

        int a = 10, b = 5;
        System.out.println("Addition: " + mo.add(a, b)); // Call the native add method
    }
}

//javac -h . MathOperations.java
//gcc -shared -o MathOperations.dll -I"C:\Program Files\Java\jdk-23\include" -I"C:\Program Files\Java\jdk-23\include\win32" MathOperations.c
//java MathOperations