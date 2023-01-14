# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

"""
Based on:
https://stackoverflow.com/questions/47310929/does-python-support-character-type
"""

def progress_bar(progress: float, 
                 total: float, 
                 prefix: str = '',
                 suffix : str = '', 
                 decimals: int = 1, 
                 length: int = 100, 
                 fill: str = '█',
                 blank: str = '-',
                 printEnd: str = "\r" ):
    
    # Determine the progress
    p = progress / total

    # Construct the bar
    fill_length = round(p*length)
    blank_length = length - fill_length
    bar = fill*fill_length + blank*blank_length
    
    # Construct the percentage
    percent_str = ('{0:.' + str(decimals) + 'f}').format(100 * p)

    # Show
    print(f'\r{prefix} |{bar}| {percent_str}% {suffix}', end = printEnd)
    
    # Print New Line on Complete
    if progress >= total: 
        print()