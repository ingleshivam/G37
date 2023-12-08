    $(document).ready(function () {
      $("#dob").change(function () {
        var dob = $("#dob").val();
        if (dob) {
          var dobDate = new Date(dob);
          var today = new Date();

          var age = today.getFullYear() - dobDate.getFullYear();

          // Check if birthday has occurred this year
          if (today.getMonth() < dobDate.getMonth() || (today.getMonth() === dobDate.getMonth() && today.getDate() < dobDate.getDate())) {
            age--;
          }

          $("#age").val(age >= 0 ? age : ""); // Ensure age is non-negative
        } else {
          $("#age").val("");
        }
      });
    });