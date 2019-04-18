// Shorten the local section number, e.g. 4.3.2.1 -> 2.1
$(document).ready(function () {
    $('h2').each(function(){
        $(this).html($(this).html().replace(/^\d+.\d+./, ''))
    });
    $('.localtoc').each(function(){
        $(this).find('a').each(function(){
            $(this).html($(this).html().replace(/^\d+\.\d+\./, ''))
        });
    });
    $('.toctree-wrapper').each(function(){
        $(this).find('a').each(function(){
            if ($(this).text().match(/^\d+\.\d+.\d+\./) != null) {
                $(this).html($(this).html().replace(/^\d+\.\d+\./, ''))
            }
        });
    });
});

// Replace the QR code with an embeded discussion thread
$(document).ready(function () {
    var discuss_str = 'Discuss'
    $('h2').each(function(){
        if ($(this).text().indexOf("Scan the QR Code") != -1) {
            var url = $(this).find('a').attr('href');
            var tokens = url.split('/');
            var topic_id = tokens[tokens.length-1];
            $(this).html('<h2>'.concat(discuss_str).concat('</h2>'));
            $(this).parent().append('<div id="discourse-comments"></div>');

            $('a').each(function(){
                if ($(this).text().indexOf("Scan the QR Code to Discuss") != -1) {
                    $(this).text(discuss_str);
                }
            });

            $('img').each(function(){
                if ($(this).attr('src').indexOf("qr_") != -1) {
                    $(this).hide();
                }
            });

            DiscourseEmbed = { discourseUrl: 'https://discuss.mxnet.io/', topicId: topic_id };
            (function() {
                var d = document.createElement('script'); d.type = 'text/javascript';
                d.async = true;
                d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
                (document.getElementsByTagName('head')[0] ||
                 document.getElementsByTagName('body')[0]).appendChild(d);
            })();
        }
    });

    var replaced = $('body').html().replace(/Scan-the-QR-Code-to-Discuss/g, discuss_str);
    $('body').html(replaced);
});
