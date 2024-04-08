function(write_version KEY VALUE)
	set(VERSION_FILE_PATH "${CMAKE_BINARY_DIR}/versions.txt")
	file(APPEND ${VERSION_FILE_PATH} "${KEY}=${VALUE}\n")
endfunction()
