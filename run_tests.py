"""
Test Runner script to execute all test functions across the test suite.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("."))

def run_all():
    import tests.test_dna as test_dna
    import tests.test_growth as test_growth
    import tests.test_routing as test_routing
    import tests.test_memory as test_memory
    import tests.test_models as test_models
    import tests.test_encoding as test_encoding
    import tests.test_evolution as test_evolution
    import tests.test_lifecycle as test_lifecycle
    import tests.test_data_pipeline as test_data_pipeline
    import tests.test_inference_io as test_inference_io
    import tests.test_lora_cppn as test_lora_cppn

    modules = [
        test_dna,
        test_growth,
        test_routing,
        test_memory,
        test_models,
        test_encoding,
        test_evolution,
        test_lifecycle,
        test_data_pipeline,
        test_inference_io,
        test_lora_cppn,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for mod in modules:
        mod_name = mod.__name__
        print(f"\n--- Running tests in {mod_name} ---")
        for attr in dir(mod):
            if attr.startswith("test_") and callable(getattr(mod, attr)):
                fn = getattr(mod, attr)
                try:
                    fn()
                    print(f"  [PASS] {attr}")
                    total_passed += 1
                except Exception as e:
                    print(f"  [FAIL] {attr}: {e}")
                    traceback.print_exc()
                    total_failed += 1
                    failures.append((mod_name, attr, str(e)))

    print("\n=========================================")
    print(f"TOTAL: {total_passed + total_failed} | PASSED: {total_passed} | FAILED: {total_failed}")
    print("=========================================")
    if total_failed > 0:
        print("\nFailed Tests:")
        for mod_name, attr, err in failures:
            print(f" - {mod_name}.{attr}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    run_all()
