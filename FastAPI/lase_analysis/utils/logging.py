import sys
import logging

logger = logging.getLogger('lase_analysis')
handlers = []

def get_handlers():
  return handlers

def add_handler(hdlr):
  logger.addHandler(hdlr)
  handlers.append(hdlr)

def add_StreamHandler(stream):
  hdlr = logging.StreamHandler(stream)
  add_handler(hdlr)

def remove_handler(hdlr):
  logger.removeHandler(hdlr)
  handlers.remove(hdlr)

def set_level(level):
  if isinstance(level, str): level = getattr(logging, level)
  logger.setLevel(level)
