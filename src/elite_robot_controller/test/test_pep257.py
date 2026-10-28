# Copyright 2015 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    # D213 (summary from the second line) is turned off. This repository, including the operating UI,
    # writes the summary **on the first line** (D212), while the ament default
    # convention enforces the opposite, so both cannot be satisfied.
    rc = main(argv=['.', 'test', '--add-ignore', 'D213'])
    assert rc == 0, 'Found code style errors / warnings'
