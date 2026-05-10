# CMake generated Testfile for 
# Source directory: /Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c
# Build directory: /Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/build_cmake
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(othello_rules "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/build_cmake/test_othello" "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/tests/fixture_othello.bin")
set_tests_properties(othello_rules PROPERTIES  _BACKTRACE_TRIPLES "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;131;add_test;/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;0;")
add_test(tree_sanity "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/build_cmake/test_tree")
set_tests_properties(tree_sanity PROPERTIES  _BACKTRACE_TRIPLES "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;136;add_test;/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;0;")
add_test(nn_parity "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/build_cmake/test_nn" "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/tests/random_model.onnx" "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/tests/nn_fixture.bin")
set_tests_properties(nn_parity PROPERTIES  _BACKTRACE_TRIPLES "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;140;add_test;/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;0;")
add_test(mcts_serial_parity "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/build_cmake/test_mcts" "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/tests/random_model.onnx" "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/tests/mcts_fixture.bin")
set_tests_properties(mcts_serial_parity PROPERTIES  _BACKTRACE_TRIPLES "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;147;add_test;/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;0;")
add_test(mcts_threaded "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/build_cmake/test_mcts_threaded" "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/tests/random_model.onnx")
set_tests_properties(mcts_threaded PROPERTIES  _BACKTRACE_TRIPLES "/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;154;add_test;/Users/mbt/Downloads/alpha-zero-general-forked-master/othello_c/CMakeLists.txt;0;")
