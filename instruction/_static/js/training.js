$(function() {
	$("button.doc-btn").hover(function(){
		
		//移入事件
		$(this).addClass('doc-btn-hover')
	},function(){
		//移出事件
		$(this).removeClass('doc-btn-hover');
	})
	// 每页显示数
	var curNum = 8
	// 计算总数
	var all = $('.doc-article-list').children('div.doc-article-item').length;

	var list = []
	
	
	
	$('button.doc-btn').click(function() {
		$('.doc-article-item').removeClass('OUO');
		var id_val = $(this).attr('id')
		if (id_val !== 'all') {
			if ($('#all').hasClass('doc-btn-color')) {
				$('#all').removeClass('doc-btn-color').find('img').remove();
				list.splice(list.indexOf('all_exist'), 1);
				$('.doc-article-item').removeClass('all_exist');
			}
		} else {
			$('button.doc-btn-color').each(function() {
				var tag = $(this).attr('id');
				$('.' + tag).removeClass(tag + '_exist');
				list.splice(list.indexOf(tag + '_exist'), 1);
			});

			$('button.doc-btn-color').removeClass('doc-btn-color').find('img').remove();
		}
		if ($(this).hasClass('doc-btn-color')) {
			$(this).removeClass('doc-btn-color').find('img').remove();
			$('.' + id_val).removeClass(id_val + '_exist');
			list.splice(list.indexOf(id_val + '_exist'), 1);
			
		} else {
			if(id_val == 'all'){
				$(this).addClass('doc-btn-color');
				$('.' + id_val).addClass(id_val + '_exist');
				list.push(id_val + '_exist');
			}else{
				$(this).addClass('doc-btn-color').append('<img src="./_static/img/choice.png" style="position: absolute; right: -5px; top: -5px;"></img>');
				$('.' + id_val).addClass(id_val + '_exist');
				list.push(id_val + '_exist');
			}
			
		}
		
		if(list.length > 0){
			var os_list = [];
			var hardware_list = [];
			var user_list = [];
			var stage_list = [];
			var experience_list = [];
			var all_list = [];
			var hasWindows = false;
			var hasCpu = false;
			
			$('.doc-article-item').addClass('hidden');
			var str = 'OUO';
			for(var i=0;i<list.length;i++){
				if(list[i].indexOf('os') == 0){
					os_list.push(list[i]);
					if (list[i].indexOf('Windows') > -1) {
						hasWindows = true;
					}
				}else if (list[i].indexOf('hardware') == 0){
					hardware_list.push(list[i]);
					if (list[i].indexOf('CPU') > -1) {
						hasCpu = true;
					}
				}else if (list[i].indexOf('user') == 0){
					user_list.push(list[i]);
				}else if (list[i].indexOf('stage') == 0){
					stage_list.push(list[i]);
				}else if (list[i].indexOf('experience') == 0){
					experience_list.push(list[i]);
				}else{
					all_list.push(list[i]);
				}
			}

			if(!((os_list.length === 1 && hasWindows) && (hardware_list.length && !hasCpu))) {
				$('.doc-article-item').each(function(){
					var os_count = 0;
					var hardware_count = 0;
					var user_count = 0;
					var stage_count = 0;
					var experience_count = 0;
					var all_count = 0;
					if(os_list.length > 0){
						for(var i=0;i<os_list.length;i++){
							if ($(this).attr('class').replace('hidden ','').indexOf(os_list[i]) > -1){
								os_count += 1;
							}						
						}
					}else{
						os_count = 'empty';
					}
					
					if(hardware_list.length > 0){
						for(var i=0;i<hardware_list.length;i++){
							if ($(this).attr('class').replace('hidden ','').indexOf(hardware_list[i]) > -1){
								hardware_count += 1;
							}						
						}
					}else{
						hardware_count = 'empty';
					}
					
					if(user_list.length > 0){
						for(var i=0;i<user_list.length;i++){
							if ($(this).attr('class').replace('hidden ','').indexOf(user_list[i]) > -1){
								user_count += 1;
							}						
						}
					}else{
						user_count = 'empty';
					}
					
					if(stage_list.length > 0){
						for(var i=0;i<stage_list.length;i++){
							if ($(this).attr('class').replace('hidden ','').indexOf(stage_list[i]) > -1){
								stage_count += 1;
							}						
						}
					}else{
						stage_count = 'empty';
					}

					if(experience_list.length > 0){
						for(var i=0;i<experience_list.length;i++){
							if ($(this).attr('class').replace('hidden ','').indexOf(experience_list[i]) > -1){
								experience_count += 1;
							}						
						}
					}else{
						experience_count = 'empty';
					}
					
					if(all_list.length > 0){
						for(var i=0;i<all_list.length;i++){
							if ($(this).attr('class').replace('hidden ','').indexOf(all_list[i]) > -1){
								all_count += 1;
							}					
						}
					}else{
						all_count = 'empty';
					}
					
					
					if(((os_count >0 && os_count <= os_list.length) || os_count=='empty') && ((hardware_count >0 && hardware_count <= hardware_list.length) || hardware_count=='empty') && ((user_count >0 && user_count <= user_list.length) || user_count == 'empty') && ((stage_count >0 && stage_count <= stage_list.length) || stage_count == 'empty') && ((experience_count >0 && experience_count <= experience_list.length) || experience_count == 'empty')){
						$(this).removeClass('hidden').addClass(str);
					}			
				});
			}

		}else{
			$('.doc-article-item').addClass('hidden');
		}

		var hidden_num = $('.doc-article-list').children('.doc-article-item.hidden').length;
		var all_article = all - hidden_num
		// 计算总页数
		var len = Math.ceil((all - hidden_num) / curNum);
		// 生成页码
		var pageList = '<li class="disabled"><span>' + '共' + all_article + '条' +  '</span></li>' + '<li class="pre"><a href="javascript:;" aria-label="Previous"><span aria-hidden="true">&laquo;</span></a></li>';
		// 当前的索引值
		var iNum = 0;

		for (var i = 0; i < len; i++) {
			pageList += '<li class="doc-data"><a href="javascript:;">' + (i + 1) + '</a></li>'
		}
		pageList += '<li class="nex"><a href="javascript:;" aria-label="Next"><span aria-hidden="true">&raquo;</span></a></li>'
		// 首页加亮显示
		if (all_article > 0){
			$('#pageNav').html(pageList).find('li').eq(2).addClass('active');
		}else{
			$('#pageNav').html('<li class="disabled"><span>' + '共' + all_article + '条' +  '</span></li>');
		}
		
		// 标签页的点击事件
		$('#pageNav').find('li.doc-data').each(function() {
			$(this).click(function() {
				$(this).addClass('active').siblings('li').removeClass('active');
				iNum = $(this).index() - 2;
				if(iNum > 0){
					$('li.pre').removeClass('disabled');
				}else{
					$('li.pre').addClass('disabled');
				}
				if(iNum+1 == len){
					$('li.nex').addClass('disabled');
				}
				$('.doc-article-item[class*="' + str + '"]').hide();
				for (var i = (iNum * curNum); i < (iNum + 1) * curNum; i++) {
					$('div.doc-article-list').find('.doc-article-item[class*="' + str + '"]').eq(i).show()
				}

			});
		});
		if(iNum == 0){
			$('li.pre').addClass('disabled');
		}
		
		if(iNum+1 == len){
			$('li.nex').addClass('disabled');
		}
		// 向前页点击时间
		$('li.pre').click(function(){
			if(iNum > 0){
				iNum -= 1;
				if(iNum == 0){
					$(this).addClass('disabled');
				}
				$('li.nex').removeClass('disabled');
				$('#pageNav').find('li.doc-data').eq(iNum).addClass('active').siblings('li').removeClass('active');
				$('.doc-article-item[class*="' + str + '"]').hide();
				for (var i = (iNum * curNum); i < (iNum + 1) * curNum; i++) {
					$('div.doc-article-list').find('.doc-article-item[class*="' + str + '"]').eq(i).show()
				}
			}
			
		});
		
		// 向后页点击事件
		$('li.nex').click(function(){
			if(iNum+1 < len){
				iNum += 1;
				if(iNum+1 == len){
					$(this).addClass('disabled');
				}
				$('li.pre').removeClass('disabled');
				$('#pageNav').find('li.doc-data').eq(iNum).addClass('active').siblings('li').removeClass('active');
				$('.doc-article-item[class*="' + str + '"]').hide();
				for (var i = (iNum * curNum); i < (iNum + 1) * curNum; i++) {
					$('div.doc-article-list').find('.doc-article-item[class*="' + str + '"]').eq(i).show()
				}
			}
		});
		
		//  首页的显示
		$('.doc-article-item[class*="' + str + '"]').hide();
		for (var i = 0; i < curNum; i++) {
			$('div.doc-article-list').find('.doc-article-item[class*="' + str + '"]').eq(i).show();
		}

		if ($('button.doc-btn-color').length == 0) {
			$('#all').trigger('click');
		}
	});


	$('#all').trigger('click');

});
