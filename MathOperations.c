#include <jni.h>
#include "MathOperations.h" // Generated header file

// JNI implementation of the add function
JNIEXPORT jint JNICALL Java_MathOperations_add(JNIEnv *env, jobject obj, jint a, jint b)
{
    // Directly return the addition result
    return a + b;
}
