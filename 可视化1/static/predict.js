$(function() {
    // 监听表单的提交事件
    $("#iris-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 从表单中获取数据
        let feature1 = $("#feature1").val();
        let feature2 = $("#feature2").val();
        let feature3 = $("#feature3").val();
        let feature4 = $("#feature4").val();
        let feature5 = $("#feature5").val();
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_input",
            type: "POST",
            data: {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3,
                "feature4": feature4,
                "feature5": feature5
            },
            success: function(response) {
                // 在页面中显示预测结果
                let prediction = response.prediction[0];
                let resultDiv = $("#result");
                resultDiv.html("预测结果为: " + prediction);
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#result");
                resultDiv.html("发生错误: " + error);
            }
        });
    });
    // 监听文件上传表单的提交事件
    $("#file-form").submit(function(event) {
        event.preventDefault();  // 阻止表单提交
        // 获取文件数据
        let file = $("input[name='file']").prop("files")[0];
        let formData = new FormData();
        formData.append("file", file);
        // 发送POST请求到服务器
        $.ajax({
            url: "/predict_from_file",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 在页面中显示预测结果
                let resultDiv = $("#file-result");
                resultDiv.html("预测结果为: " + response.predictions.join(", "));
            },
            error: function(xhr, status, error) {
                // 显示错误信息
                let resultDiv = $("#file-result");
                resultDiv.html("发生错误: " + error);
            }
        });
    });
});