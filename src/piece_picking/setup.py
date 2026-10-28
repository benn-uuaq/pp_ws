import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'piece_picking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pp_ws',
    maintainer_email='hoonyoung323@gmail.com',
    description='Piece picking - moves the robot to the picking position from the calibration result and the vision product position',
    license='Proprietary',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'picking_gui = piece_picking.picking_gui:main',
        ],
    },
)
