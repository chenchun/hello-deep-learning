#include <torch/extension.h>

// Declare the roll_call_launcher function
void roll_call_launcher();

// Write the C++ function that we will call from Python
void roll_call_binding() {
    roll_call_launcher();
}

PYBIND11_MODULE(example_kernels, m) {
  m.def(
    "roll_call", // Name of the Python function to create
    &roll_call_binding, // Corresponding C++ function to call
    "Launches the roll_call kernel" // Docstring
  );
}