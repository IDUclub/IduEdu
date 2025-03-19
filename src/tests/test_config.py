# pylint: disable=redefined-outer-name

import pytest
from loguru import logger

from iduedu._config import Config


@pytest.fixture(scope="module")
def config_instance():
    return Config()


def test_change_logger_lvl(config_instance):

    config_instance.change_logger_lvl("DEBUG")
    assert logger.level("DEBUG").no == logger.level("DEBUG").no

    config_instance.change_logger_lvl("ERROR")
    assert logger.level("ERROR").no == logger.level("ERROR").no


def test_set_overpass_url(config_instance):

    new_url = "http://new-overpass-url.com/api/interpreter"
    config_instance.set_overpass_url(new_url)
    assert config_instance.overpass_url == new_url


def test_set_timeout(config_instance):

    config_instance.set_timeout(30)
    assert config_instance.timeout == 30

    config_instance.set_timeout(0)
    assert config_instance.timeout == 0


def test_set_enable_tqdm(config_instance):

    config_instance.set_enable_tqdm(False)
    assert config_instance.enable_tqdm_bar is False

    config_instance.set_enable_tqdm(True)
    assert config_instance.enable_tqdm_bar is True


def test_config_defaults():
    config = Config()
    assert config.overpass_url == "http://lz4.overpass-api.de/api/interpreter"
    assert config.timeout is None
    assert config.enable_tqdm_bar is True
    assert config.logger == logger
