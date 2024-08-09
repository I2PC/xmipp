#***************************************************************************
# Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307  USA
#
#  All comments concerning this program package may be sent to the
#  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************

function(link_to_scipion INSTALL_DIRECTORY SCIPION_SOFTWARE)
	set(SCIPION_SOFTWARE_XMIPP ${SCIPION_SOFTWARE}/em/xmipp)

	# Link installation
	file(
		CREATE_LINK
			${INSTALL_DIRECTORY}/
			${SCIPION_SOFTWARE_XMIPP}
		COPY_ON_ERROR
		SYMBOLIC
	)

	# Link python binding
	file(GLOB PYTHON_DIR_CONTENT ${SCIPION_SOFTWARE_XMIPP}/bindings/python/*)
	foreach(x IN LISTS PYTHON_DIR_CONTENT)
		get_filename_component(y ${x} NAME)
		file(
			CREATE_LINK
				${x}
				${SCIPION_SOFTWARE}/bindings/${y}
			COPY_ON_ERROR
			SYMBOLIC
		)
	endforeach()
	
	# Link shared libraries
	set(
		LIBRARIES 
			libcifpp.so
			libcifpp.so.3
			libcifpp.so.5.0.9
			libsvm.so
			libXmippCore.so
			libXmipp.so
	)
	foreach(x IN LISTS LIBRARIES)
		file(
			CREATE_LINK
				${SCIPION_SOFTWARE_XMIPP}/lib/${x}
				${SCIPION_SOFTWARE}/lib/${x}
			COPY_ON_ERROR
			SYMBOLIC
		)
	endforeach()
endfunction()