$(function() {
    // 监听表单的提交事件
    $("#iris-form1").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 从表单中获取数据
        let feature12 = $("#feature12").val();
        let feature22 = $("#feature22").val();
        let feature32 = $("#feature32").val();
        let feature42 = $("#feature42").val();
        let feature52 = $("#feature52").val();
        let feature62 = $("#feature62").val();
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_input2",
            type: "POST",
            data: {
                "feature12": feature12,
                "feature22": feature22,
                "feature32": feature32,
                "feature42": feature42,
                "feature52": feature52,
                "feature62": feature62
            },
            success: function(response) {
                // 在页面中显示预测结果
                let prediction = response.prediction[0];
                let resultDiv = $("#result1");
                resultDiv.html("预测结果为: " + prediction);
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#result1");
                resultDiv.html("发生错误: " + error);
            }
        });
    });
    // 监听文件上传表单的提交事件
    $("#file-form1").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 获取文件数据
        let file1 = $("input[name='file1']").prop("files")[0];
        let formData1 = new FormData();
        formData1.append("file1", file1);
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_file2",
            type: "POST",
            data: formData1,
            contentType: false,
            processData: false,
            success: function(response) {
                // 在页面中显示预测结果
                let resultDiv = $("#file-result1");
                resultDiv.html("预测结果为: " + response.predictions.join(", "));
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#file-result1");
                resultDiv.html("发生错误: " + error);
            }
        });
    });
});