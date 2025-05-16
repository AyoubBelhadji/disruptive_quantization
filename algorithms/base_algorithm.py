#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from abc import ABC, abstractmethod

class AbstractAlgorithm(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def run(self, data):
        pass
