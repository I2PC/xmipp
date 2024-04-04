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
	# Copy installation
	install(
		DIRECTORY
			${INSTALL_DIRECTORY}/
		DESTINATION
			${SCIPION_SOFTWARE}/em/xmipp
	)

	# Copy installation
	install(
		DIRECTORY
			${INSTALL_DIRECTORY}/bindings/python/
		DESTINATION
			${SCIPION_SOFTWARE}/bindings
	)

	# Copy shared libraries
	install(
		FILES
			${INSTALL_DIRECTORY}/lib/libcifpp.so
			${INSTALL_DIRECTORY}/lib/libcifpp.so.3
			${INSTALL_DIRECTORY}/lib/libcifpp.so.5.0.9
			${INSTALL_DIRECTORY}/lib/libsvm.so
			${INSTALL_DIRECTORY}/lib/libXmippCore.so
			${INSTALL_DIRECTORY}/lib/libXmipp.so
		DESTINATION
			${SCIPION_SOFTWARE}/lib
	)
endfunction()
