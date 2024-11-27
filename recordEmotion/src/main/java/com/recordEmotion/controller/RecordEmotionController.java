package com.recordEmotion.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.servlet.ModelAndView;

@Controller
public class RecordEmotionController {
	
	@RequestMapping(value="/main", method=RequestMethod.GET)
	public ModelAndView showMainPage() {
		ModelAndView mav = new ModelAndView();
		mav.setViewName("main.html");
		return mav;
	}
	
	@RequestMapping(value="/calendar", method=RequestMethod.GET)
	public ModelAndView showCalendarPage() {
		ModelAndView mav = new ModelAndView();
		mav.setViewName("index.html");
		return mav;
	}
}
