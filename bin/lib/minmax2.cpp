#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <climits>
using namespace std;

int main() {
    srand(time(0));
    int n = 1000000;
    vector<int> arr(n);
    for (int i = 0; i < n; i++) arr[i] = rand() % 10000 + 1;

    int mn = INT_MAX, mx = INT_MIN;
    long long sum = 0;
    double avg = 0.0;
    double t1, t2;

    // Sequential
    t1 = omp_get_wtime();
    int smn = INT_MAX, smx = INT_MIN;
    long long ssum = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < smn) smn = arr[i];
        if (arr[i] > smx) smx = arr[i];
        ssum += arr[i];
    }
    double savg = (double)ssum / n;
    t2 = omp_get_wtime();
    cout << "=== Sequential ===" << endl;
    cout << "Min: " << smn << " | Max: " << smx << " | Sum: " << ssum << " | Avg: " << savg << endl;
    cout << "Time: " << t2 - t1 << "s" << endl << endl;

    // Parallel Reduction
    t1 = omp_get_wtime();
    #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
        sum += arr[i];
    }
    avg = (double)sum / n;
    t2 = omp_get_wtime();
    cout << "=== Parallel Reduction ===" << endl;
    cout << "Min: " << mn << " | Max: " << mx << " | Sum: " << sum << " | Avg: " << avg << endl;
    cout << "Time: " << t2 - t1 << "s" << endl;

    return 0;
}



// g++ -fopenmp minmax2.cpp -o minmax2.exe
// ./minmax2.exe