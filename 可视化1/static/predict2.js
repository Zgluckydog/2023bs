$(function() {
    // 监听表单的提交事件
    $("#iris-form2").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 从表单中获取数据
        let feature13 = $("#feature13").val();
        let feature23 = $("#feature23").val();
        let feature33 = $("#feature33").val();
        let feature43 = $("#feature43").val();
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_input3",
            type: "POST",
            data: {
                "feature13": feature13,
                "feature23": feature23,
                "feature33": feature33,
                "feature43": feature43,
            },
            success: function(response) {
                // 在页面中显示预测结果
                let prediction = response.prediction[0];
                let resultDiv = $("#result2");
                resultDiv.html("预测结果为: " + prediction);
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#result2");
                resultDiv.html("发生错误: " + error);
            }
        });
    });
    // 监听文件上传表单的提交事件
    $("#file-form2").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 获取文件数据
        let file2 = $("input[name='file2']").prop("files")[0];
        let formData = new FormData();
        formData.append("file2", file2);
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_file3",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 在页面中显示预测结果
                let resultDiv = $("#file-result2");
                resultDiv.html("预测结果为: " + response.predictions.join(", "));
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#file-result2");
                resultDiv.html("发生错误: " + error);
            }
        });
    });
});